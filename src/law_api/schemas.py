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
    chunk_id: str
    article_number: int
    citation: str
    source_page: int
    chunk_text: str
    score: float


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SearchResponse(BaseModel):
    document_id: int
    query: str
    results: list[SourceChunk]


class AskResponse(BaseModel):
    document_id: int
    question: str
    answer: str
    sources: list[SourceChunk]
    llm_model: str
    embedding_model: str
    storage_provider: str