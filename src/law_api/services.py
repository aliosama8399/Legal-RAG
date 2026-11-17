import asyncio
import hashlib
import json
import logging
import re
import shutil
import uuid
from collections.abc import Callable
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from data.prepare_law import prepare

from .config import Settings
from .exceptions import ResourceNotFoundError
from .guardrails.live_eval import DEFAULT_METRICS, evaluate_live
from .guardrails.pii import StreamRedactor, get_detector
from .helpers.config import PROJECT_ROOT
from .models.schemes import LawChunk, RetrievedDocument
from .stores.embeddings.EmbeddingInterface import EmbeddingInterface
from .stores.llm.LLMInterface import LLMInterface
from .stores.rerankers.RerankerInterface import RerankerInterface
from .stores.vectordb.providers.QdrantProvider import (
    _ARTICLE_QUERY_RE,
    _bilingual_searchable_text,
    tokenize_content,
)
from .stores.vectordb.VectorDBInterface import VectorDBInterface
from .tracking.langfuse_tracker import LangfuseTracker
from .tracking.mlflow_tracker import MLflowTracker
from .tracking.prometheus_metrics import (
    ASK_LATENCY,
    EMPTY_ANSWERS,
    FIRST_TOKEN_LATENCY,
    LLM_CALLS,
    PII_REDACTIONS,
    PII_SCANNED,
    REQUESTS,
    RETRIEVAL_RERANKED,
    RETRIEVAL_SOURCES,
    STREAM_LATENCY,
    record_live_eval,
    record_usage,
)

# How often the streaming generation span pushes its partial answer. Every
# token would flood the ingestion pipeline; every 20th keeps the UI live.
_STREAM_TRACE_EVERY = 20

logger = logging.getLogger(__name__)

_CRIMINAL_SCOPE_RE = re.compile(
    r"(?:"
    r"\b(?:penal\s+code|criminal\s+(?:law|code|liability|penalty|offen[cs]e|sentence)|"
    r"capital\s+punishment|death\s+penalty|theft|stealing|larceny|robbery|burglary|"
    r"embezzlement|murder|homicide|manslaughter|felony|misdemeanor|imprisonment|"
    r"prison|jail|punishment\s+for\s+(?:theft|murder|robbery|arson)|"
    r"penalty\s+for\s+(?:theft|murder|robbery|arson))\b"
    r"|قانون\s+العقوبات|عقوبة\s+السرقة|عقوبة\s+القتل|عقوبة\s+السجن|عقوبة\s+الحبس|"
    r"جريمة\s+السرقة|جريمة\s+القتل|السجن\s+المشدد|القتل\s+العمد|جناية|جنحة|الإعدام"
    r")",
    re.IGNORECASE,
)

_PENAL_EVIDENCE_RE = re.compile(
    r"(?:\b(?:penal\s+code|imprisonment|prison|jail|death\s+penalty|capital\s+punishment|"
    r"felony|misdemeanor|criminal\s+offen[cs]e)\b|"
    r"قانون\s+العقوبات|السجن|الحبس|الإعدام|جناية|جنحة)",
    re.IGNORECASE,
)


def _matches_criminal_scope(question: str) -> bool:
    return bool(_CRIMINAL_SCOPE_RE.search(question))


def _bilingual_chunk_text(chunk: dict) -> str:
    """Return combined Arabic + English statutory text when both exist on a chunk."""
    primary = (chunk.get("chunk_text") or "").strip()
    english = (chunk.get("text_en") or "").strip()
    arabic = (chunk.get("text_ar") or "").strip()
    parts = [part for part in (primary, english) if part]
    if arabic and arabic not in parts:
        parts.insert(0, arabic)
    # Deduplicate while preserving order
    unique = list(dict.fromkeys(parts))
    return "\n".join(unique)


def _is_out_of_scope_query(
    question: str,
    results: list[dict],
    *,
    top_dense_score: float | None = None,
) -> bool:
    """Decline criminal/penal questions or completely ungrounded non-civil queries."""
    if _matches_criminal_scope(question):
        for chunk in results:
            text = _bilingual_searchable_text(chunk)
            if _PENAL_EVIDENCE_RE.search(text):
                return False
        return True

    if _ARTICLE_QUERY_RE.search(question):
        return False

    query_tokens = set(tokenize_content(question))
    if len(query_tokens) < 2 or not results:
        return False

    corpus_tokens: set[str] = set()
    for chunk in results:
        corpus_tokens.update(tokenize_content(_bilingual_searchable_text(chunk)))

    if query_tokens & corpus_tokens:
        return False

    return top_dense_score is None or top_dense_score < 0.25


def _no_results_answer(question: str, *, out_of_scope: bool = False) -> str:
    if out_of_scope or _matches_criminal_scope(question):
        return (
            "No relevant articles were found for this question in the indexed "
            "Civil Code corpus (out of scope)."
        )
    return "No relevant articles were found for this question."


def _fuse_rerank_rrf(
    candidates: list[dict], rerank_scores: list[float], *, k: int = 60
) -> list[dict]:
    """Blend pre-rerank hybrid rank and cross-encoder rank via weighted RRF."""
    rerank_order = sorted(
        range(len(candidates)), key=lambda idx: rerank_scores[idx], reverse=True
    )
    rerank_rank = {idx: rank for rank, idx in enumerate(rerank_order, start=1)}
    fused: list[dict] = []
    for pos, chunk in enumerate(candidates):
        rrf_rank = pos + 1
        blended = 0.55 / (k + rrf_rank) + 0.45 / (k + rerank_rank[pos])
        fused.append({**chunk, "score": round(blended, 6)})
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused


def _fuse_rrf(
    dense_hits: list[dict],
    lexical_hits: list[dict],
    *,
    question: str,
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion over dense vector hits and BM25/citation lexical hits."""
    if not lexical_hits:
        return dense_hits

    allow_repealed = any(term in question.lower() for term in ("repeal", "abolish", "ملغ", "ألغي"))
    scores: dict[tuple[int, str], float] = {}
    items: dict[tuple[int, str], dict] = {}

    for rank, item in enumerate(dense_hits, start=1):
        key = (int(item["document_id"]), str(item["chunk_id"]))
        items[key] = item
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    for rank, item in enumerate(lexical_hits, start=1):
        key = (int(item["document_id"]), str(item["chunk_id"]))
        if key not in items:
            items[key] = item
        scores[key] = scores.get(key, 0.0) + 1.15 / (k + rank)

    fused: list[dict] = []
    for key, rrf_score in scores.items():
        item = dict(items[key])
        if item.get("is_repealed") and not allow_repealed:
            rrf_score *= 0.45
        item["score"] = round(rrf_score, 6)
        fused.append(item)

    fused.sort(key=lambda doc: doc["score"], reverse=True)
    return fused


class DocumentIngestionService:
    def __init__(
        self,
        settings: Settings,
        storage: VectorDBInterface,
        embedder: EmbeddingInterface,
        tracker: MLflowTracker | None = None,
        tenant_store=None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.embedder = embedder
        self.tracker = tracker
        self.tenant_store = tenant_store

    async def upload_and_chunk(
        self, pdf_path: Path, filename: str, tenant_id: str = "org_default"
    ) -> dict:
        # Isolate extracted artifacts by content hash so concurrent uploads
        # never overwrite a shared processed/ directory.
        source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        output_dir = pdf_path.parent / "processed" / source_hash
        marker = output_dir / ".source_hash"
        articles_path = output_dir / "law_articles.jsonl"
        chunk_path = output_dir / "law_chunks.jsonl"
        if not (
            marker.is_file()
            and marker.read_text(encoding="utf-8").strip() == source_hash
            and articles_path.is_file()
            and chunk_path.is_file()
        ):
            canonical_pdf = PROJECT_ROOT / "src" / "data" / "raw" / "egyptian_civil_code.pdf"
            canonical_dir = PROJECT_ROOT / "src" / "data" / "processed"
            canonical_articles = canonical_dir / "law_articles.jsonl"
            canonical_chunks = canonical_dir / "law_chunks.jsonl"
            reused_canonical = False
            if (
                canonical_pdf.is_file()
                and canonical_articles.is_file()
                and canonical_chunks.is_file()
            ):
                canonical_hash = await asyncio.to_thread(
                    lambda: hashlib.sha256(canonical_pdf.read_bytes()).hexdigest()
                )
                if canonical_hash == source_hash:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(shutil.copy2, canonical_articles, articles_path)
                    await asyncio.to_thread(shutil.copy2, canonical_chunks, chunk_path)
                    reused_canonical = True
            if not reused_canonical:
                await asyncio.to_thread(prepare, pdf_path, output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(source_hash, encoding="utf-8")

        articles = await asyncio.to_thread(self._read_jsonl, articles_path)
        chunks = await asyncio.to_thread(self._read_jsonl, chunk_path)
        validated_chunks = [LawChunk.model_validate(chunk).model_dump() for chunk in chunks]

        pending_ids = [
            int(path.stem)
            for path in self.settings.pending_dir.glob("*.json")
            if path.stem.isdigit()
        ]
        next_id_floor = (max(pending_ids) + 1) if pending_ids else 1
        document_id = await self.storage.save_pending_document(
            filename,
            str(pdf_path),
            len(articles),
            validated_chunks,
            next_id_floor,
            tenant_id=tenant_id,
        )
        pending_path = self.settings.pending_dir / f"{document_id}.json"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_payload = {"tenant_id": tenant_id, "chunks": validated_chunks}
        pending_path.write_text(
            json.dumps(pending_payload, ensure_ascii=False), encoding="utf-8"
        )
        if self.tenant_store is not None:
            self.tenant_store.upsert_document_meta(
                tenant_id=tenant_id,
                document_id=document_id,
                filename=filename,
                source_hash=source_hash,
                article_count=len(articles),
                chunk_count=len(validated_chunks),
                status="pending",
            )
        return {
            "document_id": document_id,
            "filename": filename,
            "articles": len(articles),
            "chunks": len(validated_chunks),
        }

    async def embed_and_store(
        self,
        document_id: int,
        tenant_id: str = "org_default",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        pending_path = self.settings.pending_dir / f"{document_id}.json"
        if not pending_path.exists():
            raise ValueError(f"No pending chunks found for document {document_id}")
        raw_payload = await asyncio.to_thread(
            lambda: json.loads(pending_path.read_text(encoding="utf-8"))
        )
        if isinstance(raw_payload, dict) and "chunks" in raw_payload:
            owner_tenant = raw_payload.get("tenant_id", "org_default")
            if owner_tenant != tenant_id:
                raise ValueError(f"No pending chunks found for document {document_id}")
            raw_chunks = raw_payload["chunks"]
        else:
            raw_chunks = raw_payload

        chunks = [LawChunk.model_validate(chunk).model_dump() for chunk in raw_chunks]
        texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch_vecs = await self.embedder.encode(texts[start : start + batch_size])
            embeddings.extend(batch_vecs)
            if progress_callback is not None:
                progress_callback(len(embeddings), len(texts))

        stored_chunks, dimension = await self.storage.save_embeddings(
            document_id, embeddings, self.embedder.model_id, chunks, tenant_id=tenant_id
        )
        if self.tracker is not None:
            tracker = self.tracker

            def _log_mlflow_run() -> None:
                with tracker.run(
                    model_name=self.embedder.name,
                    storage_name=self.storage.name,
                    chunk_count=len(chunks),
                ) as mlflow:
                    if mlflow:
                        mlflow.log_metrics(
                            {"embedding_dimension": dimension, "embedded_chunks": len(embeddings)}
                        )

            await asyncio.to_thread(_log_mlflow_run)
        if self.tenant_store is not None:
            self.tenant_store.upsert_document_meta(
                tenant_id=tenant_id,
                document_id=document_id,
                filename="",
                chunk_count=stored_chunks,
                status="ready",
                embedding_model=self.embedder.model_id,
            )
        return {
            "document_id": document_id,
            "chunks": stored_chunks,
            "embedded_chunks": len(embeddings),
            "embedding_dimension": dimension,
            "embedding_model": self.embedder.model_id,
            "storage_provider": self.storage.name,
        }

    async def list_documents(self, tenant_id: str = "org_default") -> list[dict]:
        if self.tenant_store is not None:
            return self.tenant_store.list_documents_meta(tenant_id)
        return await self.storage.list_documents(tenant_id=tenant_id)

    async def delete_document(self, document_id: int, tenant_id: str = "org_default") -> bool:
        deleted_storage = await self.storage.delete_document(document_id, tenant_id=tenant_id)
        deleted_meta = (
            self.tenant_store.delete_document_meta(tenant_id, document_id)
            if self.tenant_store is not None
            else False
        )
        if not (deleted_storage or deleted_meta):
            raise ResourceNotFoundError(f"Document {document_id} was not found")
        pending_path = self.settings.pending_dir / f"{document_id}.json"
        pending_path.unlink(missing_ok=True)
        return True

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RAGQueryService:
    """Hybrid retrieval and citation-grounded answer generation over tenant-isolated documents."""

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
        eval_judge_base_url: str = "",
        eval_judge_model: str = "",
        eval_judge_provider: str = "vllm",
        eval_judge_api_key: str = "",
        live_eval_path: str = "",
        tenant_store=None,
        hybrid_search: bool = True,
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
        self.eval_judge_base_url = eval_judge_base_url
        self.eval_judge_model = eval_judge_model
        self.eval_judge_provider = eval_judge_provider
        self.eval_judge_api_key = eval_judge_api_key
        self.live_eval_path = live_eval_path
        self.tenant_store = tenant_store
        self.hybrid_search = hybrid_search

    @staticmethod
    def _validated_results(raw_results: list[dict]) -> list[dict]:
        return [
            RetrievedDocument.model_validate(item).model_dump()
            for item in raw_results
            if item.get("chunk_text", "").strip()
        ]

    async def _retrieve_with_status(
        self,
        question: str,
        top_k: int | None,
        document_id: int | None,
        tenant_id: str = "org_default",
    ) -> tuple[list[float], list[dict], bool]:
        """Hybrid dense + BM25 lexical search fused via RRF, followed by cross-encoder reranking."""
        limit = top_k or self.top_k
        query_vector = (await self.embedder.encode([question]))[0]
        fetch = max(limit, self.rerank_candidates) if (self.reranker or self.hybrid_search) else limit

        dense_raw = await self.storage.search(
            document_id, query_vector, fetch, tenant_id=tenant_id
        )
        dense_hits = self._validated_results(dense_raw)
        top_dense_score = float(dense_hits[0]["score"]) if dense_hits else 0.0

        candidates = dense_hits
        if self.hybrid_search:
            try:
                lexical_raw = await self.storage.lexical_search(
                    document_id, question, fetch, tenant_id=tenant_id
                )
                lexical_hits = self._validated_results(lexical_raw)
                candidates = _fuse_rrf(dense_hits, lexical_hits, question=question)
            except Exception as err:
                logger.debug("Lexical search skipped due to error: %s", err)

        if self.reranker and len(candidates) > 1:
            documents = [_bilingual_chunk_text(chunk) for chunk in candidates]
            scores = await self.reranker.rerank(question, documents)
            candidates = _fuse_rerank_rrf(candidates, scores)

        top_results = candidates[:limit]
        if _is_out_of_scope_query(
            question, candidates[:max(limit, 5)], top_dense_score=top_dense_score
        ):
            return query_vector, [], True
        return query_vector, top_results, False

    async def _retrieve(
        self,
        question: str,
        top_k: int | None,
        document_id: int | None,
        tenant_id: str = "org_default",
    ) -> tuple[list[float], list[dict]]:
        query_vector, top_results, _ = await self._retrieve_with_status(
            question, top_k, document_id, tenant_id=tenant_id
        )
        return query_vector, top_results

    async def search(
        self,
        question: str,
        top_k: int | None = None,
        document_id: int | None = None,
        tenant_id: str = "org_default",
    ) -> list[dict]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        REQUESTS.labels(endpoint="search", mode="sync").inc()
        _, results = await self._retrieve(question, top_k, document_id, tenant_id=tenant_id)
        RETRIEVAL_SOURCES.observe(len(results))
        if self.reranker is not None:
            RETRIEVAL_RERANKED.inc()
        return results

    def _record_generation_usage(
        self,
        *,
        tenant_id: str = "org_default",
        user_id: str = "usr_default",
        operation: str = "rag_ask",
        prompt: str = "",
        answer: str = "",
        request_id: str = "",
    ) -> None:
        usage = getattr(self.llm, "last_usage", None)
        LLM_CALLS.labels(model=self.llm.model_id).inc()
        if usage:
            prompt_tokens = int(usage.get("prompt", 0))
            completion_tokens = int(usage.get("completion", 0))
            record_usage(self.llm.model_id, prompt_tokens, completion_tokens)
        else:
            prompt_tokens = max(1, len(prompt.split()))
            completion_tokens = max(1, len(answer.split()))

        if self.tenant_store is not None:
            self.tenant_store.record_usage(
                tenant_id=tenant_id,
                user_id=user_id,
                operation=operation,
                model=self.llm.model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                request_id=request_id,
            )

    async def ask(
        self,
        question: str,
        top_k: int | None = None,
        document_id: int | None = None,
        evaluate: bool = False,
        expected_article: int | None = None,
        eval_metrics: list[str] | None = None,
        tenant_id: str = "org_default",
        user_id: str = "usr_default",
        request_id: str = "",
    ) -> dict:
        if not question.strip():
            raise ValueError("Question must not be empty")
        REQUESTS.labels(endpoint="ask", mode="sync").inc()
        started = perf_counter()

        with self._span(
            "rag-ask",
            as_type="span",
            question=question,
            document_id=document_id,
            tenant_id=tenant_id,
            llm_model=self.llm.model_id,
            embedding_model=self.embedder.model_id,
            storage_provider=self.storage.name,
            top_k=top_k or self.top_k,
        ) as span:
            trace_id = (
                getattr(span, "trace_id", "") if span is not None else ""
            ) or f"tr_{uuid.uuid4().hex[:16]}"
            query_vector, results, out_of_scope = await self._retrieve_traced(
                question, top_k, document_id, tenant_id=tenant_id
            )
            base = {
                "question": question,
                "sources": results,
                "llm_model": self.llm.model_id,
                "embedding_model": self.embedder.model_id,
                "storage_provider": self.storage.name,
                "trace_id": trace_id,
            }
            if not results:
                answer = _no_results_answer(question, out_of_scope=out_of_scope)
                EMPTY_ANSWERS.inc()
            else:
                answer, prompt_text = await self._generate_traced(results, question)
                self._record_generation_usage(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    operation="rag_ask",
                    prompt=prompt_text,
                    answer=answer,
                    request_id=request_id,
                )
            answer, _redactions = self._guard_answer(answer)
            await self.storage.save_chat_message(
                question, answer, results, document_id, query_vector, tenant_id=tenant_id
            )
            RETRIEVAL_SOURCES.observe(len(results))
            if self.reranker is not None:
                RETRIEVAL_RERANKED.inc()
            ASK_LATENCY.observe(perf_counter() - started)
            if span is not None:
                span.update(
                    output={
                        "answer": answer,
                        "sources": [chunk["citation"] for chunk in results],
                        "latency_seconds": round(perf_counter() - started, 3),
                    }
                )
        self._flush()
        result = {**base, "answer": answer}
        if evaluate or expected_article is not None:
            chosen = eval_metrics or (DEFAULT_METRICS if evaluate else None)
            scores = await self._evaluate_live(
                question, answer, results, top_k or self.top_k, expected_article, chosen
            )
            result["live_eval"] = scores
            self._attach_live_scores(scores, trace_id)
        return result

    async def ask_in_conversation(
        self,
        *,
        conversation_id: str,
        question: str,
        top_k: int = 5,
        document_id: int | None = None,
        evaluate: bool = False,
        expected_article: int | None = None,
        tenant_id: str = "org_default",
        user_id: str = "usr_default",
        request_id: str = "",
    ) -> dict:
        """Multi-turn contextual RAG turn persisted to a tenant-scoped conversation."""
        if self.tenant_store is None:
            raise RuntimeError("TenantStore is required for conversations")
        conv = self.tenant_store.get_conversation(tenant_id, conversation_id)
        if conv is None:
            raise ResourceNotFoundError(f"Conversation '{conversation_id}' was not found")

        effective_doc_id = document_id if document_id is not None else conv.get("document_id")
        prior_messages = conv.get("messages", [])
        retrieval_query = self._contextualize_query(question, prior_messages)

        user_msg = self.tenant_store.append_conversation_message(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            role="user",
            content=question,
        )

        trace_id = f"tr_{uuid.uuid4().hex[:16]}"
        query_vector, results, out_of_scope = await self._retrieve_traced(
            retrieval_query, top_k, effective_doc_id, tenant_id=tenant_id
        )
        if not results:
            answer = _no_results_answer(question, out_of_scope=out_of_scope)
            EMPTY_ANSWERS.inc()
        else:
            prompt = self._build_conversation_prompt(results, prior_messages, question)
            answer = await self.llm.generate(
                prompt,
                system=self._SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self._record_generation_usage(
                tenant_id=tenant_id,
                user_id=user_id,
                operation="conversation_turn",
                prompt=prompt,
                answer=answer,
                request_id=request_id,
            )
        answer, _ = self._guard_answer(answer)

        assistant_msg = self.tenant_store.append_conversation_message(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            role="assistant",
            content=answer,
            sources=results,
            trace_id=trace_id,
        )
        await self.storage.save_chat_message(
            question, answer, results, effective_doc_id, query_vector, tenant_id=tenant_id
        )
        turn_payload = {
            "conversation_id": conversation_id,
            "user_message": user_msg,
            "assistant_message": assistant_msg,
            "sources": results,
            "llm_model": self.llm.model_id,
            "embedding_model": self.embedder.model_id,
        }
        if evaluate or expected_article is not None:
            chosen = DEFAULT_METRICS if evaluate else None
            scores = await self._evaluate_live(
                question, answer, results, top_k or self.top_k, expected_article, chosen
            )
            turn_payload["live_eval"] = scores
            self._attach_live_scores(scores, trace_id)
        return turn_payload

    @staticmethod
    def _contextualize_query(question: str, prior_messages: list[dict]) -> str:
        """Carry context from previous user turns when follow-up is short or referential."""
        user_turns = [m["content"] for m in prior_messages if m.get("role") == "user"]
        if not user_turns:
            return question
        lowered = question.lower()
        referential = any(
            tok in lowered.split()
            for tok in ("it", "that", "this", "they", "those", "thereof", "therein")
        )
        if referential or len(question.split()) <= 5:
            return f"{user_turns[-1]} {question}"
        return question

    @staticmethod
    def _build_conversation_prompt(
        results: list[dict], prior_messages: list[dict], question: str
    ) -> str:
        context = "\n\n".join(
            f"[{chunk['citation']}] {_bilingual_chunk_text(chunk)}" for chunk in results
        )
        recent = prior_messages[-4:]
        history_block = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )
        if history_block:
            return (
                f"Articles:\n{context}\n\n"
                f"Conversation History:\n{history_block}\n\n"
                f"Question: {question}\nAnswer:"
            )
        return f"Articles:\n{context}\n\nQuestion: {question}\nAnswer:"

    def _attach_live_scores(self, scores: dict, trace_id: str | None) -> None:
        if self.langfuse is None or not trace_id:
            return
        client = getattr(self.langfuse, "_client", None)
        if client is None:
            return
        for name in (
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
            "hit_at_k",
            "reciprocal_rank",
        ):
            value = scores.get(name)
            if not isinstance(value, (int, float)):
                continue
            with suppress(Exception):
                client.create_score(trace_id=trace_id, name=name, value=float(value))
        self._flush()

    def _guard_answer(self, answer: str) -> tuple[str, dict[str, int]]:
        result = get_detector().redact(answer)
        PII_SCANNED.inc()
        for entity, count in result.entities.items():
            PII_REDACTIONS.labels(entity_type=entity).inc(count)
        return result.text, result.entities

    async def _evaluate_live(
        self,
        question: str,
        answer: str,
        results: list[dict],
        top_k: int,
        expected_article: int | None,
        metrics: list[str] | None,
    ) -> dict:
        payload = await evaluate_live(
            question=question,
            answer=answer,
            results=results,
            top_k=top_k,
            expected_article=expected_article,
            metrics=tuple(metrics) if metrics else None,
            judge_base_url=self.eval_judge_base_url,
            judge_model=self.eval_judge_model,
            judge_provider=self.eval_judge_provider,
            judge_api_key=self.eval_judge_api_key,
        )
        record_live_eval(payload, payload.get("duration_seconds") or 0.0)
        self._persist_live_eval(payload)
        return payload

    def _persist_live_eval(self, payload: dict) -> None:
        if not self.live_eval_path:
            return
        try:
            path = Path(self.live_eval_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            body = {
                **payload,
                "question_preview": payload.get("question_preview", ""),
                "embedding_model": self.embedder.model_id,
                "generation_model": self.llm.model_id,
                "top_k": payload.get("top_k"),
                "written_at": datetime.now(UTC).isoformat(),
            }
            path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        except Exception as error:
            logger.warning(
                "could not persist live evaluation to %s (%s); the Grafana live "
                "panels will reset on restart",
                self.live_eval_path,
                error,
            )

    def _span(self, name: str, as_type: str = "span", **attributes):
        return (
            self.langfuse.trace(name, as_type=as_type, **attributes)
            if self.langfuse
            else nullcontext()
        )

    def _flush(self) -> None:
        if self.langfuse is not None:
            self.langfuse.flush()

    async def _retrieve_traced(
        self,
        question: str,
        top_k: int | None,
        document_id: int | None,
        tenant_id: str = "org_default",
    ) -> tuple[list[float], list[dict], bool]:
        with self._span(
            "vector-search",
            as_type="retriever",
            question=question,
            top_k=top_k or self.top_k,
            rerank=self.reranker is not None,
            embedding_model=self.embedder.model_id,
        ) as span:
            query_vector, results, out_of_scope = await self._retrieve_with_status(
                question, top_k, document_id, tenant_id=tenant_id
            )
            if span is not None:
                span.update(
                    output={
                        "sources": [chunk["citation"] for chunk in results],
                        "returned": len(results),
                        "out_of_scope": out_of_scope,
                    }
                )
        return query_vector, results, out_of_scope

    async def _generate_traced(self, results: list[dict], question: str) -> tuple[str, str]:
        prompt = self._build_prompt(results, question)
        with self._span(
            "answer-generation",
            as_type="generation",
            model=self.llm.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            prompt=prompt,
        ) as span:
            answer = await self.llm.generate(
                prompt,
                system=self._SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if span is not None:
                span.update(output={"answer": answer})
        return answer, prompt

    _SYSTEM_PROMPT = (
        "You are a helpful legal research assistant for the Egyptian Civil Code. "
        "Answer the user's question directly and concisely in 2-4 sentences based strictly on the provided articles, "
        "quoting the relevant article text when it answers the question. "
        "Always cite article numbers inline like (Article 147). "
        "Do not extrapolate, invent laws, or answer criminal/penal code questions from civil articles. "
        "If the provided articles do not contain the answer or the question is outside the Civil Code, say so in one sentence and decline to answer."
    )

    @staticmethod
    def _build_prompt(results: list[dict], question: str) -> str:
        context = "\n\n".join(
            f"[{chunk['citation']}] {_bilingual_chunk_text(chunk)}" for chunk in results
        )
        return f"Articles:\n{context}\n\nQuestion: {question}\nAnswer:"

    async def ask_stream(
        self,
        question: str,
        top_k: int | None = None,
        document_id: int | None = None,
        evaluate: bool = False,
        expected_article: int | None = None,
        eval_metrics: list[str] | None = None,
        tenant_id: str = "org_default",
        user_id: str = "usr_default",
        request_id: str = "",
    ):
        if not question.strip():
            raise ValueError("Question must not be empty")
        REQUESTS.labels(endpoint="ask", mode="stream").inc()
        started = perf_counter()
        trace_id = f"tr_{uuid.uuid4().hex[:16]}"
        with self._span(
            "rag-ask",
            as_type="span",
            question=question,
            document_id=document_id,
            tenant_id=tenant_id,
            llm_model=self.llm.model_id,
            embedding_model=self.embedder.model_id,
            storage_provider=self.storage.name,
            top_k=top_k or self.top_k,
            stream=True,
        ) as root_span:
            if root_span is not None and getattr(root_span, "trace_id", ""):
                trace_id = root_span.trace_id
            query_vector, results, out_of_scope = await self._retrieve_traced(
                question, top_k, document_id, tenant_id=tenant_id
            )
            yield {"type": "sources", "question": question, "sources": results}

            if not results:
                answer = _no_results_answer(question, out_of_scope=out_of_scope)
                EMPTY_ANSWERS.inc()
            else:
                prompt = self._build_prompt(results, question)
                with self._span(
                    "answer-generation",
                    as_type="generation",
                    model=self.llm.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    prompt=prompt,
                ) as gen_span:
                    redactor = StreamRedactor(get_detector())
                    collected: list[str] = []
                    stream_started = perf_counter()
                    source = self.llm.generate_stream(
                        prompt,
                        system=self._SYSTEM_PROMPT,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    async for chunk in redactor.guarded(source):
                        if not collected:
                            FIRST_TOKEN_LATENCY.observe(perf_counter() - stream_started)
                        collected.append(chunk)
                        yield {"type": "token", "text": chunk}
                        if gen_span is not None and len(collected) % _STREAM_TRACE_EVERY == 0:
                            gen_span.update(output={"partial_answer": "".join(collected)})
                    answer = "".join(collected)
                    if redactor.entities:
                        PII_SCANNED.inc()
                        for entity, count in redactor.entities.items():
                            PII_REDACTIONS.labels(entity_type=entity).inc(count)
                    if gen_span is not None:
                        gen_span.update(output={"answer": answer})
                self._record_generation_usage(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    operation="rag_ask_stream",
                    prompt=prompt,
                    answer=answer,
                    request_id=request_id,
                )

            answer, _ = self._guard_answer(answer)

            await self.storage.save_chat_message(
                question, answer, results, document_id, query_vector, tenant_id=tenant_id
            )
            RETRIEVAL_SOURCES.observe(len(results))
            if self.reranker is not None:
                RETRIEVAL_RERANKED.inc()
            STREAM_LATENCY.observe(perf_counter() - started)
            if root_span is not None:
                root_span.update(
                    output={
                        "answer": answer,
                        "sources": [chunk["citation"] for chunk in results],
                        "latency_seconds": round(perf_counter() - started, 3),
                    }
                )
        self._flush()
        yield {"type": "done", "answer": answer, "sources": results, "trace_id": trace_id}

        if evaluate or expected_article is not None:
            chosen = eval_metrics or (DEFAULT_METRICS if evaluate else None)
            scores = await self._evaluate_live(
                question, answer, results, top_k or self.top_k, expected_article, chosen
            )
            self._attach_live_scores(scores, trace_id)
            yield {"type": "eval", **scores}

    async def history(
        self,
        document_id: int | None = None,
        limit: int = 50,
        tenant_id: str = "org_default",
    ) -> list[dict]:
        return await self.storage.list_chat_history(
            document_id, limit, tenant_id=tenant_id
        )
