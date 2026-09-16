import asyncio
import json
from contextlib import nullcontext
from pathlib import Path

from data.prepare_law import prepare

from .config import Settings
from .models.schemes import LawChunk, RetrievedDocument
from .stores.embeddings.EmbeddingInterface import EmbeddingInterface
from .stores.llm.LLMInterface import LLMInterface
from .stores.vectordb.VectorDBInterface import VectorDBInterface
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
        # pdfplumber extraction is blocking CPU work — run it off the event loop.
        output_dir = pdf_path.parent / "processed"
        article_path, chunk_path = await asyncio.to_thread(prepare, pdf_path, output_dir)
        articles = await asyncio.to_thread(self._read_jsonl, article_path)
        chunks = await asyncio.to_thread(self._read_jsonl, chunk_path)
        # Validate every chunk through pydantic before it is stored.
        validated_chunks = [LawChunk.model_validate(chunk).model_dump() for chunk in chunks]
        document_id = await self.storage.save_pending_document(
            filename, str(pdf_path), len(articles), validated_chunks
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
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def search(self, question: str, top_k: int | None = None, document_id: int | None = None) -> list[dict]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        query_vector = (await self.embedder.encode([question]))[0]
        results = await self.storage.search(document_id, query_vector, top_k or self.top_k)
        # Validate every retrieved chunk through pydantic before it leaves the service.
        return [RetrievedDocument.model_validate(item).model_dump() for item in results]

    async def ask(self, question: str, top_k: int | None = None, document_id: int | None = None) -> dict:
        if not question.strip():
            raise ValueError("Question must not be empty")
        query_vector = (await self.embedder.encode([question]))[0]
        results = [
            RetrievedDocument.model_validate(item).model_dump()
            for item in await self.storage.search(document_id, query_vector, top_k or self.top_k)
        ]
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

        context = "\n\n".join(f"[{chunk['citation']}] {chunk['chunk_text']}" for chunk in results)
        system = (
            "You are a legal research assistant. Answer only using the provided "
            "Egyptian Civil Code articles and cite the article number for every claim. "
            "If the answer is not contained in the provided articles, say so."
        )
        prompt = f"Articles:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = await self.llm.generate(
            prompt, system=system, temperature=self.temperature, max_tokens=self.max_tokens
        )
        await self.storage.save_chat_message(question, answer, results, document_id, query_vector)
        return {**base, "answer": answer}

    async def history(self, document_id: int | None = None, limit: int = 50) -> list[dict]:
        return await self.storage.list_chat_history(document_id, limit)
