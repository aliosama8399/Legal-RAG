import asyncio
import json
import logging
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from data.prepare_law import prepare

from .config import Settings
from .guardrails.live_eval import DEFAULT_METRICS, evaluate_live
from .guardrails.pii import StreamRedactor, get_detector
from .models.schemes import LawChunk, RetrievedDocument
from .stores.embeddings.EmbeddingInterface import EmbeddingInterface
from .stores.llm.LLMInterface import LLMInterface
from .stores.rerankers.RerankerInterface import RerankerInterface
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
        eval_judge_base_url: str = "",
        eval_judge_model: str = "",
        live_eval_path: str = "",
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
        self.live_eval_path = live_eval_path

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
        REQUESTS.labels(endpoint="search", mode="sync").inc()
        _, results = await self._retrieve(question, top_k, document_id)
        RETRIEVAL_SOURCES.observe(len(results))
        if self.reranker is not None:
            RETRIEVAL_RERANKED.inc()
        return results

    def _record_generation_usage(self) -> None:
        """Move the provider's token counts into Prometheus.

        `last_usage` is None when the backend does not report usage, which is
        not the same as zero tokens — recording 0 there would silently make the
        cost/hour panel read low instead of admitting the gap.
        """
        usage = getattr(self.llm, "last_usage", None)
        LLM_CALLS.labels(model=self.llm.model_id).inc()
        if not usage:
            return
        record_usage(
            self.llm.model_id, usage.get("prompt", 0), usage.get("completion", 0)
        )

    async def ask(
        self,
        question: str,
        top_k: int | None = None,
        document_id: int | None = None,
        evaluate: bool = False,
        expected_article: int | None = None,
        eval_metrics: list[str] | None = None,
    ) -> dict:
        if not question.strip():
            raise ValueError("Question must not be empty")
        REQUESTS.labels(endpoint="ask", mode="sync").inc()
        started = perf_counter()

        # The root span opens BEFORE the work, so the trace is already visible in
        # Langfuse while retrieval and generation are still running.
        with self._span(
            "rag-ask",
            as_type="span",
            question=question,
            document_id=document_id,
            llm_model=self.llm.model_id,
            embedding_model=self.embedder.model_id,
            storage_provider=self.storage.name,
            top_k=top_k or self.top_k,
        ) as span:
            trace_id = getattr(span, "trace_id", "") if span is not None else ""
            query_vector, results = await self._retrieve_traced(question, top_k, document_id)
            base = {
                "question": question,
                "sources": results,
                "llm_model": self.llm.model_id,
                "embedding_model": self.embedder.model_id,
                "storage_provider": self.storage.name,
            }
            if not results:
                answer = "No relevant articles were found for this question."
                EMPTY_ANSWERS.inc()
            else:
                answer = await self._generate_traced(results, question)
                self._record_generation_usage()
            answer, _redactions = self._guard_answer(answer)
            await self.storage.save_chat_message(question, answer, results, document_id, query_vector)
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
            # `evaluate: true` means "score the answer too" — an omitted
            # eval_metrics falls back to the defaults rather than silently
            # scoring nothing. Supplying only expected_article stays free,
            # because that half is arithmetic.
            chosen = eval_metrics or (DEFAULT_METRICS if evaluate else None)
            scores = await self._evaluate_live(
                question, answer, results, top_k or self.top_k, expected_article, chosen
            )
            result["live_eval"] = scores
            self._attach_live_scores(scores, trace_id)
        return result

    def _attach_live_scores(self, scores: dict, trace_id: str | None) -> None:
        """Attach live scores to the trace, so Langfuse filters on them too."""
        if self.langfuse is None or not trace_id:
            return
        client = getattr(self.langfuse, "_client", None)
        if client is None:
            return
        for name in ("faithfulness", "answer_relevancy", "context_relevancy", "hit_at_k", "reciprocal_rank"):
            value = scores.get(name)
            if not isinstance(value, (int, float)):
                continue
            try:
                client.create_score(trace_id=trace_id, name=name, value=float(value))
            except Exception:
                # Scoring is best-effort; never fail a served request over it.
                pass

    def _guard_answer(self, answer: str) -> tuple[str, dict[str, int]]:
        """Run the PII guardrail over a completed answer.

        Always recorded, even when nothing was found, so the dashboards show the
        guardrail is actually scanning rather than silently absent.
        """
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
        """Score this served request and publish it as gauges.

        Runs inline rather than in the background: BentoML drives the ASGI app
        on a per-request event loop, so a task started here never runs again.
        The cost is reported in the response instead of being hidden.
        """
        payload = await evaluate_live(
            question=question,
            answer=answer,
            results=results,
            top_k=top_k,
            expected_article=expected_article,
            metrics=tuple(metrics) if metrics else None,
            judge_base_url=self.eval_judge_base_url,
            judge_model=self.eval_judge_model,
        )
        record_live_eval(payload, payload.get("duration_seconds") or 0.0)
        self._persist_live_eval(payload)
        return payload

    def _persist_live_eval(self, payload: dict) -> None:
        """Write the scores so they survive a restart.

        Every gauge in this process is lost when the container is recreated, and
        that left the Grafana live panels reading zero after each rebuild. The
        metrics bridge republishes this file on the next scrape.
        """
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
            # The gauges are already set in-process, so a failure here is easy to
            # miss and silently costs the Grafana panels their value on the next
            # restart. Warn loudly instead: the usual cause is a read-only mount.
            logger.warning(
                "could not persist live evaluation to %s (%s); the Grafana live "
                "panels will reset on restart",
                self.live_eval_path,
                error,
            )

    def _span(self, name: str, as_type: str = "span", **attributes):
        """Live Langfuse span; yields None when tracing is off or unreachable.

        Langfuse nests observations through the active OpenTelemetry context, so
        a span opened inside another one is recorded as its child. `as_type` is
        the most specific type available for the step, which is what drives the
        observation analytics in the UI.
        """
        return (
            self.langfuse.trace(name, as_type=as_type, **attributes)
            if self.langfuse
            else nullcontext()
        )

    def _flush(self) -> None:
        if self.langfuse is not None:
            self.langfuse.flush()

    async def _retrieve_traced(
        self, question: str, top_k: int | None, document_id: int | None
    ) -> tuple[list[float], list[dict]]:
        """Retrieval in its own span, so the trace shows it while it is running."""
        with self._span(
            "vector-search",
            as_type="retriever",
            question=question,
            top_k=top_k or self.top_k,
            rerank=self.reranker is not None,
            embedding_model=self.embedder.model_id,
        ) as span:
            query_vector, results = await self._retrieve(question, top_k, document_id)
            if span is not None:
                span.update(
                    output={
                        "sources": [chunk["citation"] for chunk in results],
                        "returned": len(results),
                    }
                )
        return query_vector, results

    async def _generate_traced(self, results: list[dict], question: str) -> str:
        """Non-streaming generation in its own span."""
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
        return answer

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
        self,
        question: str,
        top_k: int | None = None,
        document_id: int | None = None,
        evaluate: bool = False,
        expected_article: int | None = None,
        eval_metrics: list[str] | None = None,
    ):
        """Async generator yielding SSE events: sources -> tokens -> done -> eval.

        The query embedding happens once (one short question string); the
        stored chunk embeddings from /embed are reused, never remade.
        """
        if not question.strip():
            raise ValueError("Question must not be empty")
        REQUESTS.labels(endpoint="ask", mode="stream").inc()
        started = perf_counter()
        # The span stays open for the whole stream, so the trace is live from the
        # first token to the last one.
        with self._span(
            "rag-ask",
            as_type="span",
            question=question,
            document_id=document_id,
            llm_model=self.llm.model_id,
            embedding_model=self.embedder.model_id,
            storage_provider=self.storage.name,
            top_k=top_k or self.top_k,
            stream=True,
        ) as span:
            query_vector, results = await self._retrieve_traced(question, top_k, document_id)
            yield {"type": "sources", "question": question, "sources": results}

            if not results:
                answer = "No relevant articles were found for this question."
                EMPTY_ANSWERS.inc()
            else:
                # The token loop lives here, not in a helper, because each
                # released chunk has to become an SSE event. It used to be
                # hidden inside _generate_traced, which returned a single
                # string: the answer was produced but never streamed, so the
                # client saw sources and then silence.
                prompt = self._build_prompt(results, question)
                with self._span(
                    "answer-generation",
                    as_type="generation",
                    model=self.llm.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    prompt=prompt,
                ) as span:
                    redactor = StreamRedactor(get_detector())
                    collected: list[str] = []
                    stream_started = perf_counter()
                    source = self.llm.generate_stream(
                        prompt,
                        system=self._SYSTEM_PROMPT,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    # Tokens are withheld until they have been scanned, so a
                    # detected entity never reaches the client mid-stream.
                    async for chunk in redactor.guarded(source):
                        if not collected:
                            FIRST_TOKEN_LATENCY.observe(perf_counter() - stream_started)
                        collected.append(chunk)
                        yield {"type": "token", "text": chunk}
                        if span is not None and len(collected) % _STREAM_TRACE_EVERY == 0:
                            span.update(output={"partial_answer": "".join(collected)})
                    answer = "".join(collected)
                    if redactor.entities:
                        PII_SCANNED.inc()
                        for entity, count in redactor.entities.items():
                            PII_REDACTIONS.labels(entity_type=entity).inc(count)
                    if span is not None:
                        span.update(output={"answer": answer})
                self._record_generation_usage()

            # The token stream was already redacted as it was produced; this is
            # the idempotent safety net for the assembled text we store.
            answer, _ = self._guard_answer(answer)

            await self.storage.save_chat_message(question, answer, results, document_id, query_vector)
            RETRIEVAL_SOURCES.observe(len(results))
            if self.reranker is not None:
                RETRIEVAL_RERANKED.inc()
            STREAM_LATENCY.observe(perf_counter() - started)
            if span is not None:
                span.update(
                    output={
                        "answer": answer,
                        "sources": [chunk["citation"] for chunk in results],
                        "latency_seconds": round(perf_counter() - started, 3),
                    }
                )
        self._flush()
        yield {"type": "done", "answer": answer, "sources": results}

        # Live evaluation is a separate trailing event so the answer streams at
        # full speed and the judge call never delays the response.
        if evaluate or expected_article is not None:
            trace_id = getattr(span, "trace_id", "") if span is not None else ""
            chosen = eval_metrics or (DEFAULT_METRICS if evaluate else None)
            scores = await self._evaluate_live(
                question, answer, results, top_k or self.top_k, expected_article, chosen
            )
            self._attach_live_scores(scores, trace_id)
            yield {"type": "eval", **scores}

    async def history(self, document_id: int | None = None, limit: int = 50) -> list[dict]:
        return await self.storage.list_chat_history(document_id, limit)
