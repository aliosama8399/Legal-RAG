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
    status: str = "ok"
    documents_indexed: int = 0
    chunks_indexed: int = 0
    models_loaded: bool = True


class ErrorResponse(BaseModel):
    detail: str


class SourceChunk(BaseModel):
    document_id: int
    chunk_id: str
    article_number: int
    citation: str
    source_page: int
    chunk_text: str
    text_ar: str = ""
    text_en: str = ""
    score: float


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
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
    judge_provider: str | None = None
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
    trace_id: str | None = None


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


# ---- Enterprise SaaS Schemas ----


class RegisterRequest(BaseModel):
    org_name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(default="Workspace Owner", min_length=1, max_length=120)
    plan: str = Field(default="pro")


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class UserProfileResponse(BaseModel):
    tenant_id: str
    org_name: str
    user_id: str
    email: str
    role: str
    plan: str
    auth_type: str


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    profile: UserProfileResponse


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    scopes: list[str] = Field(default_factory=lambda: ["read", "write"])


class ApiKeyCreatedResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    api_key: str
    scopes: list[str]
    created_at: str


class ApiKeyItem(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    active: bool
    revoked_at: str | None = None
    last_used_at: str | None = None
    created_at: str


class ApiKeyListResponse(BaseModel):
    results: list[ApiKeyItem]


class ModelUsageBreakdown(BaseModel):
    model: str
    calls: int
    tokens: int
    cost_usd: float


class UsageSummaryResponse(BaseModel):
    tenant_id: str
    org_name: str
    plan: str
    total_operations: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    monthly_token_quota: int
    quota_utilization_pct: float
    estimated_cost_usd: float
    documents_count: int
    max_documents: int
    by_model: list[ModelUsageBreakdown]


class AuditLogEntry(BaseModel):
    id: str
    tenant_id: str
    actor_id: str
    action: str
    resource_type: str
    resource_id: str
    metadata: dict
    ip_address: str
    request_id: str
    created_at: str


class AuditLogResponse(BaseModel):
    results: list[AuditLogEntry]


class DocumentSummary(BaseModel):
    document_id: int
    tenant_id: str
    filename: str
    article_count: int
    chunk_count: int
    status: str
    embedding_model: str
    created_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    results: list[DocumentSummary]


class IngestionJobResponse(BaseModel):
    id: str
    tenant_id: str
    document_id: int | None = None
    filename: str
    status: str
    stage: str
    progress_pct: int
    articles: int = 0
    chunks: int = 0
    error: str | None = None
    created_at: str
    updated_at: str


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New Legal Research Thread", max_length=200)
    document_id: int | None = None


class ConversationSummary(BaseModel):
    id: str
    tenant_id: str
    user_id: str
    document_id: int | None = None
    title: str
    message_count: int = 0
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    results: list[ConversationSummary]


class ConversationMessageRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: int | None = None
    evaluate: bool = False
    expected_article: int | None = None


class ConversationMessageEntry(BaseModel):
    id: str
    role: str
    content: str
    sources: list[dict] = Field(default_factory=list)
    trace_id: str = ""
    created_at: str


class ConversationDetailResponse(BaseModel):
    id: str
    tenant_id: str
    user_id: str
    document_id: int | None = None
    title: str
    created_at: str
    updated_at: str
    messages: list[ConversationMessageEntry]


class ConversationTurnResponse(BaseModel):
    conversation_id: str
    user_message: ConversationMessageEntry
    assistant_message: ConversationMessageEntry
    sources: list[SourceChunk]
    llm_model: str
    embedding_model: str
    live_eval: LiveEvalScores | None = None


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1)
    score: int = Field(ge=0, le=1)
    message_id: str = ""
    comment: str = Field(default="", max_length=1000)


class FeedbackResponse(BaseModel):
    id: str
    tenant_id: str
    trace_id: str
    score: int
    comment: str
    created_at: str