export interface Source {
  chunk_id?: string
  chunk_text: string
  citation: string
  score?: number
  article_number?: number | string
  document_id?: number
  source_page?: number
  text_ar?: string
  text_en?: string
}

export interface AskResponse {
  question: string
  answer: string
  sources: Source[]
  llm_model: string
  embedding_model: string
  storage_provider: string
  trace_id?: string | null
}

export interface LiveEvalScores {
  hit_at_k: number | null
  reciprocal_rank: number | null
  retrieved_ranks: number[]
  faithfulness: number | null
  answer_relevancy: number | null
  context_relevancy: number | null
  judge_model: string | null
  duration_seconds: number | null
  degraded: string | null
}

export type StreamEvent =
  | { type: 'sources'; question: string; sources: Source[] }
  | { type: 'token'; text: string }
  | { type: 'done'; answer: string; sources: Source[]; trace_id?: string }
  | ({ type: 'eval' } & LiveEvalScores)
  | { type: 'error'; detail?: string; error?: string }

export interface HistoryItem {
  id?: number
  document_id?: number | null
  question: string
  answer: string
  created_at?: string
  sources?: Source[]
}

export interface HistoryResponse {
  results: HistoryItem[]
}

export interface UploadResult {
  document_id: number
  filename: string
  articles: number
  chunks: number
}

export interface EmbedResult {
  document_id: number
  chunks: number
  embedded_chunks: number
  embedding_dimension: number
  embedding_model: string
  storage_provider: string
}

export interface UserProfile {
  tenant_id: string
  org_name: string
  user_id: string
  email: string
  role: string
  plan: string
  auth_type: string
}

export interface AuthTokenResponse {
  access_token: string
  token_type: string
  expires_in: number
  profile: UserProfile
}

export interface ApiKeyCreated {
  id: string
  name: string
  key_prefix: string
  api_key: string
  scopes: string[]
  created_at: string
}

export interface ApiKeyItem {
  id: string
  name: string
  key_prefix: string
  scopes: string[]
  active: boolean
  revoked_at?: string | null
  last_used_at?: string | null
  created_at: string
}

export interface UsageSummary {
  tenant_id: string
  org_name: string
  plan: string
  total_operations: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  monthly_token_quota: number
  quota_utilization_pct: number
  estimated_cost_usd: number
  documents_count: number
  max_documents: number
  by_model: Array<{
    model: string
    calls: number
    tokens: number
    cost_usd: number
  }>
}

export interface AuditLogEntry {
  id: string
  tenant_id: string
  actor_id: string
  action: string
  resource_type: string
  resource_id: string
  metadata: Record<string, unknown>
  ip_address: string
  request_id: string
  created_at: string
}

export interface DocumentSummary {
  document_id: number
  tenant_id: string
  filename: string
  article_count: number
  chunk_count: number
  status: string
  embedding_model: string
  created_at: string
  updated_at: string
}

export interface IngestionJob {
  id: string
  tenant_id: string
  document_id?: number | null
  filename: string
  status: string
  stage: string
  progress_pct: number
  articles: number
  chunks: number
  error?: string | null
  created_at: string
  updated_at: string
}

export interface ConversationSummary {
  id: string
  tenant_id: string
  user_id: string
  document_id?: number | null
  title: string
  message_count: number
  created_at: string
  updated_at: string
}

export interface ConversationMessageEntry {
  id: string
  role: string
  content: string
  sources: Source[]
  trace_id: string
  created_at: string
}

export interface ConversationDetail {
  id: string
  tenant_id: string
  user_id: string
  document_id?: number | null
  title: string
  created_at: string
  updated_at: string
  messages: ConversationMessageEntry[]
}

export interface ConversationTurnResponse {
  conversation_id: string
  user_message: ConversationMessageEntry
  assistant_message: ConversationMessageEntry
  sources: Source[]
  llm_model: string
  embedding_model: string
  live_eval?: LiveEvalScores | null
}
