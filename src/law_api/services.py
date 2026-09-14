import json
from contextlib import nullcontext
from pathlib import Path

from data.prepare_law import prepare

from .config import Settings
from .providers.embeddings.model_provider import EmbeddingProvider
from .providers.llm.base import LLMProvider
from .providers.storage.base import StorageProvider
from .tracking.mlflow_tracker import MLflowTracker


class DocumentIngestionService:
    def __init__(
        self,
        settings: Settings,
        storage: StorageProvider,
        embedder: EmbeddingProvider,
        tracker: MLflowTracker | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.embedder = embedder
        self.tracker = tracker

    def upload_and_chunk(self, pdf_path: Path, filename: str) -> dict:
        output_dir = pdf_path.parent / "processed"
        article_path, chunk_path = prepare(pdf_path, output_dir)
        articles = self._read_jsonl(article_path)
        chunks = self._read_jsonl(chunk_path)
        document_id = self.storage.save_pending_document(filename, str(pdf_path), len(articles), chunks)
        pending_path = self.settings.pending_dir / f"{document_id}.json"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        return {
            "document_id": document_id,
            "filename": filename,
            "articles": len(articles),
            "chunks": len(chunks),
        }

    def embed_and_store(self, document_id: int) -> dict:
        pending_path = self.settings.pending_dir / f"{document_id}.json"
        if not pending_path.exists():
            raise ValueError(f"No pending chunks found for document {document_id}")
        chunks = json.loads(pending_path.read_text(encoding="utf-8"))
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
            embeddings = self.embedder.encode([chunk["chunk_text"] for chunk in chunks])
            stored_chunks, dimension = self.storage.save_embeddings(
                document_id, embeddings, self.embedder.model_id
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
        storage: StorageProvider,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
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

    def search(self, document_id: int, question: str, top_k: int | None = None) -> list[dict]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        query_vector = self.embedder.encode([question])[0]
        return self.storage.search(document_id, query_vector, top_k or self.top_k)

    def ask(self, document_id: int, question: str, top_k: int | None = None) -> dict:
        results = self.search(document_id, question, top_k)
        base = {
            "document_id": document_id,
            "question": question,
            "sources": results,
            "llm_model": self.llm.model_id,
            "embedding_model": self.embedder.model_id,
            "storage_provider": self.storage.name,
        }
        if not results:
            return {**base, "answer": "No relevant articles were found for this question."}

        context = "\n\n".join(f"[{chunk['citation']}] {chunk['chunk_text']}" for chunk in results)
        system = (
            "You are a legal research assistant. Answer only using the provided "
            "Egyptian Civil Code articles and cite the article number for every claim. "
            "If the answer is not contained in the provided articles, say so."
        )
        prompt = f"Articles:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = self.llm.generate(
            prompt, system=system, temperature=self.temperature, max_tokens=self.max_tokens
        )
        return {**base, "answer": answer}
