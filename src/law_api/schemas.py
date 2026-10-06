from pydantic import BaseModel, Field


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
    # Live evaluation, opt-in per request because it costs a judge call.
    evaluate: bool = False
    # Gold article for the retrieval half of the live score. When supplied, the
    # retrieval metrics (hit@k, MRR, reciprocal rank) are computed without any
    # LLM call at all; when absent, only the answer-side metrics are produced.
    expected_article: int | None = None
    # Which answer-side metrics to run. Deliberately short by default: an LLM
    # judge costs real time, and faithfulness is the one that matters most.
    eval_metrics: list[str] | None = None


class LiveEvalScores(BaseModel):
    """Scores for one served request. Retrieval metrics need no judge call."""

    hit_at_k: float | None = None
    reciprocal_rank: float | None = None
    retrieved_ranks: list[int] = Field(default_factory=list)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_relevancy: float | None = None
    judge_model: str | None = None
    duration_seconds: float | None = None
    degraded: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    llm_model: str
    embedding_model: str
    storage_provider: str
    live_eval: LiveEvalScores | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SourceChunk]


class ChatHistoryEntry(BaseModel):
    id: int
    document_id: int | None
    question: str
    answer: str
    sources: list[dict]
    created_at: str


class ChatHistoryResponse(BaseModel):
    results: list[ChatHistoryEntry]