import type {
  ApiKeyCreated,
  ApiKeyItem,
  AskResponse,
  AuditLogEntry,
  AuthTokenResponse,
  ConversationDetail,
  ConversationSummary,
  ConversationTurnResponse,
  DocumentSummary,
  EmbedResult,
  HistoryItem,
  HistoryResponse,
  IngestionJob,
  LiveEvalScores,
  Source,
  StreamEvent,
  UploadResult,
  UsageSummary,
  UserProfile,
} from './types'

const BASE = '/api/v1'
const TOKEN_KEY = 'legal_rag_jwt'
const API_KEY_STORAGE = 'legal_rag_api_key'

export function setAuthToken(token: string | null): void {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token)
    localStorage.removeItem(API_KEY_STORAGE)
  } else {
    localStorage.removeItem(TOKEN_KEY)
  }
}

export function setApiKeyHeader(apiKey: string | null): void {
  if (apiKey) {
    localStorage.setItem(API_KEY_STORAGE, apiKey)
    localStorage.removeItem(TOKEN_KEY)
  } else {
    localStorage.removeItem(API_KEY_STORAGE)
  }
}

export function clearAuth(): void {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(API_KEY_STORAGE)
}

export function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem(TOKEN_KEY)
  if (token) return { Authorization: `Bearer ${token}` }
  const apiKey = localStorage.getItem(API_KEY_STORAGE)
  if (apiKey) return { 'X-API-Key': apiKey }
  return {}
}

async function jsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      }
    } catch {
      // keep HTTP status line
    }
    throw new Error(detail)
  }
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = {
    ...getAuthHeaders(),
    ...(init.headers as Record<string, string> | undefined),
  }
  return fetch(`${BASE}${path}`, { ...init, headers }).then((res) => jsonOrThrow<T>(res))
}

// ---- Auth & Tenant Admin ----

export function registerOrg(payload: {
  org_name: string
  email: string
  password: string
  full_name?: string
  plan?: string
}): Promise<AuthTokenResponse> {
  return apiFetch<AuthTokenResponse>('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function loginUser(email: string, password: string): Promise<AuthTokenResponse> {
  return apiFetch<AuthTokenResponse>('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
}

export function getMe(): Promise<UserProfile> {
  return apiFetch<UserProfile>('/auth/me')
}

export function listApiKeys(): Promise<ApiKeyItem[]> {
  return apiFetch<{ results: ApiKeyItem[] }>('/tenants/api-keys').then((r) => r.results ?? [])
}

export function createApiKey(name: string): Promise<ApiKeyCreated> {
  return apiFetch<ApiKeyCreated>('/tenants/api-keys', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, scopes: ['read', 'write'] }),
  })
}

export function revokeApiKey(keyId: string): Promise<void> {
  return apiFetch<void>(`/tenants/api-keys/${encodeURIComponent(keyId)}`, { method: 'DELETE' })
}

export function getTenantUsage(): Promise<UsageSummary> {
  return apiFetch<UsageSummary>('/tenants/usage')
}

export function getAuditLogs(limit = 50): Promise<AuditLogEntry[]> {
  return apiFetch<{ results: AuditLogEntry[] }>(`/tenants/audit-logs?limit=${limit}`).then(
    (r) => r.results ?? [],
  )
}

// ---- Documents & Async Ingestion ----

export function listDocuments(): Promise<DocumentSummary[]> {
  return apiFetch<{ results: DocumentSummary[] }>('/documents').then((r) => r.results ?? [])
}

export function ingestDocumentAsync(file: File, idempotencyKey?: string): Promise<IngestionJob> {
  const form = new FormData()
  form.append('file', file)
  const headers: Record<string, string> = {}
  if (idempotencyKey) headers['Idempotency-Key'] = idempotencyKey
  return apiFetch<IngestionJob>('/documents/ingest', {
    method: 'POST',
    headers,
    body: form,
  })
}

export function getIngestionJob(jobId: string): Promise<IngestionJob> {
  return apiFetch<IngestionJob>(`/documents/jobs/${encodeURIComponent(jobId)}`)
}

export function deleteDocument(documentId: number): Promise<void> {
  return apiFetch<void>(`/documents/${documentId}`, { method: 'DELETE' })
}

export function upload(file: File): Promise<UploadResult> {
  const form = new FormData()
  form.append('file', file)
  return apiFetch<UploadResult>('/documents/upload', { method: 'POST', body: form })
}

export function embed(documentId: number): Promise<EmbedResult> {
  return apiFetch<EmbedResult>(`/documents/${documentId}/embed`, { method: 'POST' })
}

// ---- RAG, Multi-Turn Conversations & Feedback ----

export function ask(question: string, topK: number, documentId?: number): Promise<AskResponse> {
  return apiFetch<AskResponse>('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK, document_id: documentId ?? null }),
  })
}

export function search(question: string, topK: number, documentId?: number): Promise<Source[]> {
  return apiFetch<{ results: Source[] }>('/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK, document_id: documentId ?? null }),
  }).then((body) => body.results ?? [])
}

export function history(limit = 20): Promise<HistoryItem[]> {
  return apiFetch<HistoryResponse>(`/history?limit=${limit}`).then((body) => body?.results ?? [])
}

export function listConversations(): Promise<ConversationSummary[]> {
  return apiFetch<{ results: ConversationSummary[] }>('/conversations').then(
    (r) => r.results ?? [],
  )
}

export function createConversation(
  title: string,
  documentId?: number | null,
): Promise<ConversationSummary> {
  return apiFetch<ConversationSummary>('/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, document_id: documentId ?? null }),
  })
}

export function getConversation(conversationId: string): Promise<ConversationDetail> {
  return apiFetch<ConversationDetail>(`/conversations/${encodeURIComponent(conversationId)}`)
}

export function sendConversationMessage(
  conversationId: string,
  question: string,
  topK = 5,
  documentId?: number | null,
  options: { evaluate?: boolean; expectedArticle?: number | null } = {},
): Promise<ConversationTurnResponse> {
  return apiFetch<ConversationTurnResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        top_k: topK,
        document_id: documentId ?? null,
        evaluate: options.evaluate ?? false,
        expected_article: options.expectedArticle ?? null,
      }),
    },
  )
}

export function deleteConversation(conversationId: string): Promise<void> {
  return apiFetch<void>(`/conversations/${encodeURIComponent(conversationId)}`, {
    method: 'DELETE',
  })
}

export function submitFeedback(
  traceId: string,
  score: 0 | 1,
  messageId = '',
  comment = '',
): Promise<void> {
  return apiFetch<void>('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trace_id: traceId, score, message_id: messageId, comment }),
  })
}

export async function askStream(
  question: string,
  topK: number,
  handlers: {
    onSources?: (sources: Source[]) => void
    onToken?: (token: string) => void
    onDone?: (answer: string, traceId?: string) => void
    onEval?: (scores: LiveEvalScores) => void
  },
  options: {
    signal?: AbortSignal
    evaluate?: boolean
    expectedArticle?: number | null
    documentId?: number | null
  } = {},
): Promise<void> {
  const { signal, evaluate = false, expectedArticle = null, documentId = null } = options
  const response = await fetch(`${BASE}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({
      question,
      top_k: topK,
      document_id: documentId,
      evaluate,
      expected_article: expectedArticle,
    }),
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`stream failed: ${response.status} ${response.statusText}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (!payload) continue
        let event: StreamEvent
        try {
          event = JSON.parse(payload) as StreamEvent
        } catch {
          continue
        }
        if (event.type === 'sources') handlers.onSources?.(event.sources)
        else if (event.type === 'token') handlers.onToken?.(event.text)
        else if (event.type === 'done') handlers.onDone?.(event.answer, event.trace_id)
        else if (event.type === 'eval') handlers.onEval?.(event)
        else if (event.type === 'error') throw new Error(event.detail ?? event.error ?? 'RAG error')
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}

export async function metricsSnapshot(): Promise<Record<string, number>> {
  const response = await fetch(`${BASE}/metrics`, { headers: getAuthHeaders() })
  if (!response.ok) throw new Error(`metrics unavailable: ${response.status}`)
  const text = await response.text()
  const wanted = [
    'rag_requests_total',
    'rag_llm_tokens_total',
    'rag_ragas_metric',
    'rag_ragas_questions',
    'rag_embedding_drift_cosine',
    'rag_embedding_drift_questions',
    'rag_embedding_drift_baseline_cosine',
    'rag_embedding_drift_delta',
    'rag_pii_redactions_total',
    'process_start_time_seconds',
  ]
  const out: Record<string, number> = {}
  for (const line of text.split('\n')) {
    if (line.startsWith('#') || !line.trim()) continue
    const [nameWithLabels, value] = line.split(/\s+/)
    if (!value) continue
    const name = nameWithLabels.replace(/\{.*\}$/, '')
    if (!wanted.includes(name)) continue
    out[nameWithLabels] = Number(value)
  }
  return out
}

export function health(): Promise<{ status?: string }> {
  return fetch(`${BASE}/health`).then((response) => jsonOrThrow(response))
}
