import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

from data.prepare_law import prepare

from .config import Settings
from .models.schemes import LawChunk, RetrievedDocument
from .stores.embeddings.EmbeddingInterface import EmbeddingInterface
from .stores.llm.LLMInterface import LLMInterface
from .stores.rerankers.RerankerInterface import RerankerInterface
from .stores.vectordb.VectorDBInterface import VectorDBInterface
from .tracking.langfuse_tracker import LangfuseTracker
from .tracking.mlflow_tracker import MLflowTracker


class DocumentIngestionService:
    def __init__(
        self,
        settings: Settings,
        storage: VectorDBInterface,
        embedder: EmbeddingInterface,
        tracker: MLflowTracker | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.embedder = embedder
        self.tracker = tracker

    async def upload_and_chunk(self, pdf_path: Path, filename: str) -> dict:
        # pdfplumber extraction is blocking CPU work — run it off the event loop,
        # and skip it entirely when this exact PDF was already processed
        # (uploads are content-hash named; parsing the 170-page law PDF is slow).
        output_dir = pdf_path.parent / "processed"
        import hashlib

        source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        marker = output_dir / ".source_hash"
        articles_path = output_dir / "law_articles.jsonl"
        chunk_path = output_dir / "law_chunks.jsonl"
        if not (
            marker.is_file()
            and marker.read_text(encoding="utf-8").strip() == source_hash
            and articles_path.is_file()
            and chunk_path.is_file()
        ):
            await asyncio.to_thread(prepare, pdf_path, output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(source_hash, encoding="utf-8")
        articles = await asyncio.to_thread(self._read_jsonl, articles_path)
        chunks = await asyncio.to_thread(self._read_jsonl, chunk_path)
        # Validate every chunk through pydantic before it is stored.
        validated_chunks = [LawChunk.model_validate(chunk).model_dump() for chunk in chunks]
        # Id floor from persisted pending files: pending files survive restarts,
        # so a restart between upload and embed never reuses an id.
        pending_ids = [
            int(path.stem)
            for path in self.settings.pending_dir.glob("*.json")
            if path.stem.isdigit()
        ]
        next_id_floor = (max(pending_ids) + 1) if pending_ids else 1
        document_id = await self.storage.save_pending_document(
            filename, str(pdf_path), len(articles), validated_chunks, next_id_floor
        )
        pending_path = self.settings.pending_dir / f"{document_id}.json"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(json.dumps(validated_chunks, ensure_ascii=False), encoding="utf-8")
        return {
            "document_id": document_id,
            "filename": filename,
            "articles": len(articles),
            "chunks": len(validated_chunks),
        }

    async def embed_and_store(self, document_id: int) -> dict:
        pending_path = self.settings.pending_dir / f"{document_id}.json"
        if not pending_path.exists():
            raise ValueError(f"No pending chunks found for document {document_id}")
        raw_chunks = await asyncio.to_thread(
            lambda: json.loads(pending_path.read_text(encoding="utf-8"))
        )
        chunks = [LawChunk.model_validate(chunk).model_dump() for chunk in raw_chunks]
        run_context = (
            self.tracker.run(
                model_name=self.embedder.name,
                storage_name=self.storage.name,
                chunk_count=len(chunks),
            )
            if self.tracker
            else nullcontext()
        )
        with run_context as mlflow:
            embeddings = await self.embedder.encode([chunk["chunk_text"] for chunk in chunks])
            stored_chunks, dimension = await self.storage.save_embeddings(
                document_id, embeddings, self.embedder.model_id, chunks
            )
            if mlflow:
                mlflow.log_metrics(
                    {"embedding_dimension": dimension, "embedded_chunks": len(embeddings)}
                )
        return {
            "document_id": document_id,
            "chunks": stored_chunks,
            "embedded_chunks": len(embeddings),
            "embedding_dimension": dimension,
            "embedding_model": self.embedder.model_id,
            "storage_provider": self.storage.name,
        }

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RAGQueryService:
    """Retrieval and citation-grounded answer generation over an embedded document."""

    def __init__(
        self,
        storage: VectorDBInterface,
        embedder: EmbeddingInterface,
        llm: LLMInterface,
        top_k: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 512,
        langfuse: LangfuseTracker | None = None,
        reranker: RerankerInterface | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.langfuse = langfuse
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    @staticmethod
    def _validated_results(raw_results: list[dict]) -> list[dict]:
        # Older stores may hold chunks with empty text (placeholder articles);
        # never return them as search hits.
        return [
            RetrievedDocument.model_validate(item).model_dump()
            for item in raw_results
            if item.get("chunk_text", "").strip()
        ]

    async def _retrieve(
        self, question: str, top_k: int | None, document_id: int | None
    ) -> tuple[list[float], list[dict]]:
        """Vector search, then cross-encoder reranking over a larger candidate set."""
        limit = top_k or self.top_k
        query_vector = (await self.embedder.encode([question]))[0]
        fetch = max(limit, self.rerank_candidates) if self.reranker else limit
        candidates = self._validated_results(
            await self.storage.search(document_id, query_vector, fetch)
        )
        if self.reranker and len(candidates) > limit:
            documents = [chunk["chunk_text"] for chunk in candidates]
            scores = await self.reranker.rerank(question, documents)
            candidates = [
                {**chunk, "score": float(score)}
                for chunk, score in sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
            ]
        return query_vector, candidates[:limit]

    async def search(self, question: str, top_k: int | None = None, document_id: int | None = None) -> list[dict]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        _, results = await self._retrieve(question, top_k, document_id)
        return results

    async def ask(self, question: str, top_k: int | None = None, document_id: int | None = None) -> dict:
        if not question.strip():
            raise ValueError("Question must not be empty")
        started = perf_counter()
        query_vector, results = await self._retrieve(question, top_k, document_id)
        base = {
            "question": question,
            "sources": results,
            "llm_model": self.llm.model_id,
            "embedding_model": self.embedder.model_id,
            "storage_provider": self.storage.name,
        }
        if not results:
            answer = "No relevant articles were found for this question."
            await self.storage.save_chat_message(question, answer, results, document_id, query_vector)
            return {**base, "answer": answer}

        answer = await self.llm.generate(
            self._build_prompt(results, question),
            system=self._SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        await self.storage.save_chat_message(question, answer, results, document_id, query_vector)
        self._trace_ask(question, results, answer, started)
        return {**base, "answer": answer}

    def _trace_ask(self, question: str, results: list[dict], answer: str, started: float) -> None:
        if self.langfuse is None:
            return
        with self.langfuse.trace(
            "rag-ask",
            question=question,
            llm_model=self.llm.model_id,
            embedding_model=self.embedder.model_id,
            sources=[chunk["citation"] for chunk in results],
            latency_seconds=round(perf_counter() - started, 3),
        ) as observation:
            if observation is not None:
                observation.update(output={"answer": answer})

    _SYSTEM_PROMPT = (
        "You are a helpful legal research assistant for the Egyptian Civil Code. "
        "Answer the user's question directly and concisely in 2-4 sentences, "
        "quoting the relevant article text when it answers the question. "
        "Always cite article numbers inline like (Article 147). "
        "If the provided articles do not contain the answer, say so in one sentence."
    )

    @staticmethod
    def _build_prompt(results: list[dict], question: str) -> str:
        context = "\n\n".join(f"[{chunk['citation']}] {chunk['chunk_text']}" for chunk in results)
        return f"Articles:\n{context}\n\nQuestion: {question}\nAnswer:"

    async def ask_stream(
        self, question: str, top_k: int | None = None, document_id: int | None = None
    ):
        """Async generator yielding SSE events: sources -> tokens -> done.

        The query embedding happens once (one short question string); the
        stored chunk embeddings from /embed are reused, never remade.
        """
        if not question.strip():
            raise ValueError("Question must not be empty")
        started = perf_counter()
        query_vector, results = await self._retrieve(question, top_k, document_id)
        yield {"type": "sources", "question": question, "sources": results}

        if not results:
            answer = "No relevant articles were found for this question."
            await self.storage.save_chat_message(question, answer, results, document_id, query_vector)
            yield {"type": "done", "answer": answer, "sources": results}
            return

        collected: list[str] = []
        async for token in self.llm.generate_stream(
            self._build_prompt(results, question),
            system=self._SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            collected.append(token)
            yield {"type": "token", "text": token}

        answer = "".join(collected)
        await self.storage.save_chat_message(question, answer, results, document_id, query_vector)
        self._trace_ask(question, results, answer, started)
        yield {"type": "done", "answer": answer, "sources": results}

    async def history(self, document_id: int | None = None, limit: int = 50) -> list[dict]:
        return await self.storage.list_chat_history(document_id, limit)
