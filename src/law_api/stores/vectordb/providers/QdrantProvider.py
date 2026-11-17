import math
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from qdrant_client import AsyncQdrantClient, models

from ..VectorDBInterface import VectorDBInterface

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
_ARTICLE_QUERY_RE = re.compile(r"(?:article|art\.?|مادة|المادة)\s*(\d+)", re.IGNORECASE)
_AR_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0640]")
_STOPWORDS = frozenset(
    {
        "what",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "for",
        "in",
        "to",
        "on",
        "by",
        "with",
        "does",
        "do",
        "did",
        "can",
        "could",
        "would",
        "should",
        "say",
        "about",
        "how",
        "when",
        "where",
        "who",
        "which",
        "that",
        "this",
        "these",
        "those",
        "if",
        "or",
        "and",
        "from",
        "at",
        "as",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "must",
        "may",
        "will",
        "shall",
        "ما",
        "ماهو",
        "ماهي",
        "حكم",
        "هل",
        "في",
        "فى",
        "من",
        "على",
        "عن",
        "إلى",
        "الى",
        "أو",
        "او",
        "أن",
        "ان",
        "لا",
        "الذي",
        "التي",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "كان",
        "تكون",
        "يكون",
    }
)


def _normalize_arabic(token: str) -> str:
    token = _AR_DIACRITICS_RE.sub("", token)
    token = token.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    if token.endswith("ة") and len(token) > 2:
        token = f"{token[:-1]}ه"
    if token.endswith("ى") and len(token) > 2:
        token = f"{token[:-1]}ي"
    return token


def tokenize_content(text: str) -> list[str]:
    """Bilingual normalized tokenization preserving Arabic stems and English inflections."""
    raw = [
        tok.lower()
        for tok in _TOKEN_RE.findall(text or "")
        if len(tok) > 1 or tok.isdigit()
    ]
    tokens: list[str] = []
    for tok in raw:
        if tok in _STOPWORDS:
            continue
        norm = _normalize_arabic(tok)
        if norm in _STOPWORDS:
            continue
        tokens.append(norm)
        if norm.startswith("ال") and len(norm) > 4:
            tokens.append(norm[2:])
        elif len(norm) > 4 and norm.isascii():
            if norm.endswith("ies"):
                tokens.append(f"{norm[:-3]}y")
            elif norm.endswith("es") and len(norm) > 5:
                tokens.append(norm[:-2])
            elif norm.endswith("s") and not norm.endswith("ss"):
                tokens.append(norm[:-1])
            elif norm.endswith("ed") and len(norm) > 5:
                tokens.append(norm[:-2])
            elif norm.endswith("ing") and len(norm) > 6:
                tokens.append(norm[:-3])
    return tokens


def _tokenize(text: str) -> list[str]:
    return tokenize_content(text)


def _bilingual_searchable_text(payload: dict) -> str:
    parts = [payload.get("citation", ""), payload.get("topic", "")]
    chunk_text = payload.get("chunk_text", "")
    text_ar = payload.get("text_ar", "")
    text_en = payload.get("text_en", "")
    primary_ar = text_ar or chunk_text
    if primary_ar:
        parts.append(primary_ar)
    if chunk_text and chunk_text not in (text_ar, text_en):
        parts.append(chunk_text)
    if text_en and text_en != primary_ar:
        parts.append(text_en)
    return " ".join(part for part in parts if part)


def _tenant_doc_filter(tenant_id: str, document_id: int | None = None) -> models.Filter:
    must = [models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
    if document_id is not None:
        must.append(
            models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
        )
    return models.Filter(must=must)


class QdrantProvider(VectorDBInterface):
    """Async Qdrant client with strict multi-tenant payload partitioning and hybrid lexical scoring."""

    name = "qdrant"

    @staticmethod
    def _point_id(tenant_id: str, document_id: int, chunk_id: str) -> str:
        """Deterministic UUID per (tenant_id, document_id, chunk_id) for idempotent upserts."""
        return str(uuid5(NAMESPACE_URL, f"{tenant_id}:{document_id}:{chunk_id}"))

    def __init__(self, collection: str, url: str = "", path: Path | None = None) -> None:
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            self.client = AsyncQdrantClient(path=str(path))
        else:
            self.client = AsyncQdrantClient(url=url, timeout=10)
        self.collection = collection
        self.chat_history_collection = f"{collection}_chat_history"
        self.pending: dict[int, tuple[str, list[dict]]] = {}
        self.next_document_id: int = 1
        self.next_chat_id: int = 1

    async def connect(self) -> None:
        try:
            await self.client.get_collection(self.collection)
        except Exception:
            self.next_document_id = 1
            return
        points, _ = await self.client.scroll(
            collection_name=self.collection, limit=10_000, with_payload=["document_id"]
        )
        max_id = max((point.payload.get("document_id", 0) for point in points), default=0)
        self.next_document_id = max_id + 1

    async def disconnect(self) -> None:
        await self.client.close()

    async def save_pending_document(
        self,
        filename,
        source_path,
        articles,
        chunks,
        next_id_floor: int = 0,
        tenant_id: str = "org_default",
    ):
        document_id = max(self.next_document_id, next_id_floor)
        self.next_document_id = document_id + 1
        self.pending[document_id] = (tenant_id, chunks)
        return document_id

    async def get_document_chunks(self, document_id, tenant_id: str = "org_default"):
        entry = self.pending.get(document_id)
        if entry is None:
            raise ValueError(f"Document {document_id} was not found")
        owner_tenant, chunks = entry
        if owner_tenant != tenant_id:
            raise ValueError(f"Document {document_id} was not found")
        return chunks

    async def _ensure_collection(self, dimension: int) -> None:
        try:
            collection = await self.client.get_collection(self.collection)
            if collection.config.params.vectors.size == dimension:
                return
            await self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        except Exception:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    async def _index_points(
        self,
        document_id,
        filename,
        source_path,
        chunks,
        embeddings,
        embedding_model,
        dimension,
        tenant_id: str = "org_default",
    ):
        await self._ensure_collection(dimension)
        points = [
            models.PointStruct(
                id=self._point_id(tenant_id, document_id, chunk["chunk_id"]),
                vector=vector,
                payload={
                    **chunk,
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "document_filename": filename,
                    "source_path": source_path,
                    "embedding_model": embedding_model,
                },
            )
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        # Batch upserts to stay within Qdrant payload limits for large PDFs.
        batch_size = 256
        for idx in range(0, len(points), batch_size):
            await self.client.upsert(
                collection_name=self.collection, points=points[idx : idx + batch_size]
            )

    async def save_document(
        self,
        filename,
        source_path,
        articles,
        chunks,
        embeddings,
        embedding_model,
        tenant_id: str = "org_default",
    ):
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding")
        dimension = len(embeddings[0]) if embeddings else 0
        if not dimension:
            raise ValueError("Cannot index an empty embedding set")
        document_id = await self.save_pending_document(
            filename, source_path, articles, chunks, tenant_id=tenant_id
        )
        await self._index_points(
            document_id,
            filename,
            source_path,
            chunks,
            embeddings,
            embedding_model,
            dimension,
            tenant_id=tenant_id,
        )
        return document_id, dimension

    async def save_embeddings(
        self,
        document_id,
        embeddings,
        embedding_model,
        chunks=None,
        tenant_id: str = "org_default",
    ):
        if chunks is None:
            chunks = await self.get_document_chunks(document_id, tenant_id=tenant_id)
        if len(chunks) != len(embeddings):
            raise ValueError("Embedding count does not match stored chunks")
        dimension = len(embeddings[0]) if embeddings else 0
        if not dimension:
            raise ValueError("Cannot index an empty embedding set")
        await self._index_points(
            document_id,
            f"document-{document_id}",
            "",
            chunks,
            embeddings,
            embedding_model,
            dimension,
            tenant_id=tenant_id,
        )
        return len(embeddings), dimension

    async def search(self, document_id, query_vector, top_k, tenant_id: str = "org_default"):
        if not await self.client.collection_exists(self.collection):
            raise ValueError(
                f"Document {document_id} was not found"
                if document_id is not None
                else "No relevant chunks were found"
            )
        query_filter = _tenant_doc_filter(tenant_id, document_id)
        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        hits = response.points
        if not hits:
            message = (
                f"Document {document_id} was not found or has no embeddings — you must embed the file first"
                if document_id is not None
                else "No relevant chunks were found"
            )
            raise ValueError(message)
        return [{**hit.payload, "score": hit.score} for hit in hits]

    async def lexical_search(
        self,
        document_id: int | None,
        query_text: str,
        top_k: int,
        tenant_id: str = "org_default",
    ) -> list[dict]:
        """BM25 + exact article-number matching over tenant-scoped chunks."""
        if not await self.client.collection_exists(self.collection):
            return []
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []
        explicit_articles = {int(m) for m in _ARTICLE_QUERY_RE.findall(query_text)}
        query_filter = _tenant_doc_filter(tenant_id, document_id)
        points, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=2500,
            with_payload=True,
        )
        if not points:
            return []

        docs_tokens = []
        df: dict[str, int] = {}
        for pt in points:
            payload = pt.payload or {}
            text = _bilingual_searchable_text(payload)
            toks = _tokenize(text)
            docs_tokens.append((payload, toks))
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1

        n_docs = len(docs_tokens)
        avg_dl = sum(len(t) for _, t in docs_tokens) / max(1, n_docs)
        k1, b = 1.2, 0.75
        unique_query_terms = set(query_tokens)

        scored: list[tuple[float, dict]] = []
        for payload, toks in docs_tokens:
            dl = len(toks)
            tf_map: dict[str, int] = {}
            for tok in toks:
                tf_map[tok] = tf_map.get(tok, 0) + 1
            bm25 = 0.0
            for q in query_tokens:
                freq = tf_map.get(q, 0)
                if freq == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5))
                bm25 += idf * (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / max(1.0, avg_dl)))
            if bm25 > 0.0 and len(unique_query_terms) >= 2:
                opening_en = set(_tokenize((payload.get("text_en") or "")[:140]))
                opening_ar = set(
                    _tokenize((payload.get("text_ar") or payload.get("chunk_text") or "")[:140])
                )
                if unique_query_terms <= opening_en or unique_query_terms <= opening_ar:
                    bm25 += 4.0
            if explicit_articles and payload.get("article_number") in explicit_articles:
                bm25 += 25.0
            if bm25 > 0.0:
                scored.append((bm25, {**payload, "score": round(bm25, 6)}))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    async def list_documents(self, tenant_id: str = "org_default") -> list[dict]:
        if not await self.client.collection_exists(self.collection):
            return []
        points, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=_tenant_doc_filter(tenant_id),
            limit=10_000,
            with_payload=["document_id", "document_filename", "embedding_model"],
        )
        docs: dict[int, dict] = {}
        for pt in points:
            payload = pt.payload or {}
            doc_id = payload.get("document_id")
            if doc_id is None:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "filename": payload.get("document_filename", f"document-{doc_id}"),
                    "embedding_model": payload.get("embedding_model", ""),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1
        return sorted(docs.values(), key=lambda d: d["document_id"], reverse=True)

    async def delete_document(self, document_id: int, tenant_id: str = "org_default") -> bool:
        removed_pending = False
        if document_id in self.pending and self.pending[document_id][0] == tenant_id:
            del self.pending[document_id]
            removed_pending = True

        if not await self.client.collection_exists(self.collection):
            return removed_pending

        doc_filter = _tenant_doc_filter(tenant_id, document_id)
        existing, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=doc_filter,
            limit=1,
            with_payload=False,
        )
        if not existing and not removed_pending:
            return False

        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=doc_filter),
        )
        return True

    async def _ensure_chat_history_collection(self, dimension: int) -> None:
        try:
            collection = await self.client.get_collection(self.chat_history_collection)
            if collection.config.params.vectors.size == dimension:
                return
        except Exception:
            await self.client.create_collection(
                collection_name=self.chat_history_collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    async def save_chat_message(
        self,
        question,
        answer,
        sources,
        document_id,
        query_vector,
        tenant_id: str = "org_default",
    ):
        await self._ensure_chat_history_collection(len(query_vector))
        chat_id = self.next_chat_id
        self.next_chat_id += 1
        point = models.PointStruct(
            id=str(uuid4()),
            vector=query_vector,
            payload={
                "chat_id": chat_id,
                "tenant_id": tenant_id,
                "document_id": document_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        await self.client.upsert(collection_name=self.chat_history_collection, points=[point])
        return chat_id

    async def list_chat_history(
        self, document_id=None, limit=50, tenant_id: str = "org_default"
    ):
        query_filter = _tenant_doc_filter(tenant_id, document_id)
        try:
            points, _ = await self.client.scroll(
                collection_name=self.chat_history_collection,
                scroll_filter=query_filter,
                limit=limit,
            )
        except Exception:
            return []
        entries = []
        for idx, point in enumerate(points, start=1):
            payload = point.payload or {}
            numeric_id = payload.get("chat_id")
            if not isinstance(numeric_id, int):
                numeric_id = point.id if isinstance(point.id, int) else idx
            entries.append({"id": numeric_id, **payload})
        entries.sort(key=lambda entry: entry["created_at"], reverse=True)
        return entries[:limit]
