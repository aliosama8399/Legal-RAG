from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: int
    filename: str
    articles: int
    chunks: int


class EmbedResponse(BaseModel):
    document_id: int
    chunks: int
    embedded_chunks: int
    embedding_dimension: int
    embedding_model: str
    storage_provider: str


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str


class SourceChunk(BaseModel):
    document_id: int
    chunk_id: str
    article_number: int
    citation: str
    source_page: int
    chunk_text: str
    score: float


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    document_id: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SourceChunk]


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    llm_model: str
    embedding_model: str
    storage_provider: str


class ChatHistoryEntry(BaseModel):
    id: int
    document_id: int | None
    question: str
    answer: str
    sources: list[dict]
    created_at: str


class ChatHistoryResponse(BaseModel):
    results: list[ChatHistoryEntry]