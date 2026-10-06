import type {
  AskResponse,
  LiveEvalScores,
  EmbedResult,
  HistoryItem,
  HistoryResponse,
  Source,
  StreamEvent,
  UploadResult,
} from './types'

const BASE = '/api/v1'

async function jsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      // keep the status line
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export function ask(question: string, topK: number, documentId?: number): Promise<AskResponse> {
  return fetch(`${BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      top_k: topK,
      document_id: documentId ?? null,
    }),
  }).then((response) => jsonOrThrow<AskResponse>(response))
}

export function search(question: string, topK: number, documentId?: number): Promise<Source[]> {
  return fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      top_k: topK,
      document_id: documentId ?? null,
    }),
  }).then((response) => jsonOrThrow<{ results: Source[] }>(response)).then((body) => body.results ?? [])
}

export function history(limit = 20): Promise<HistoryItem[]> {
  return fetch(`${BASE}/history?limit=${limit}`)
    .then((response) => jsonOrThrow<HistoryResponse>(response))
    .then((body) => body?.results ?? [])
}

export function upload(file: File): Promise<UploadResult> {
  const form = new FormData()
  form.append('file', file)
  return fetch(`${BASE}/documents/upload`, { method: 'POST', body: form }).then((response) =>
    jsonOrThrow<UploadResult>(response),
  )
}

export function embed(documentId: number): Promise<EmbedResult> {
  return fetch(`${BASE}/documents/${documentId}/embed`, { method: 'POST' }).then((response) =>
    jsonOrThrow<EmbedResult>(response),
  )
}

/**
 * Stream an answer token by token.
 *
 * Reads the SSE body as a byte stream and splits on the blank-line frame
 * boundary rather than using EventSource, because EventSource is GET-only and
 * this endpoint is a POST. onToken fires for each token, so the UI renders
 * progressively exactly like curl does.
 */
export async function askStream(
  question: string,
  topK: number,
  handlers: {
    onSources?: (sources: Source[]) => void
    onToken?: (token: string) => void
    onDone?: (answer: string) => void
    onEval?: (scores: LiveEvalScores) => void
  },
  options: { signal?: AbortSignal; evaluate?: boolean; expectedArticle?: number | null } = {},
): Promise<void> {
  const { signal, evaluate = false, expectedArticle = null } = options
  const response = await fetch(`${BASE}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      top_k: topK,
      document_id: null,
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
        else if (event.type === 'done') handlers.onDone?.(event.answer)
        else if (event.type === 'eval') handlers.onEval?.(event)
        else if (event.type === 'error') throw new Error(event.error)
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}

/** Read a counter out of the Prometheus text exposition on /api/v1/metrics. */
export async function metricsSnapshot(): Promise<Record<string, number>> {
  const response = await fetch(`${BASE}/metrics`)
  if (!response.ok) throw new Error(`metrics unavailable: ${response.status}`)
  const text = await response.text()
  const wanted = [
    'rag_requests_total',
    'rag_llm_tokens_total',
    'rag_ragas_metric',
    'rag_ragas_questions',
    'rag_embedding_drift_cosine',
    'rag_embedding_drift_questions',
    // Drift is judged as a drop against the baseline's own mean; the raw cosine
    // alone is not a health signal.
    'rag_embedding_drift_baseline_cosine',
    'rag_embedding_drift_delta',
    'rag_pii_redactions_total',
    // Makes the counter reset visible: every rag_* counter restarts with the
    // process, and "0 requests" otherwise looks like a broken pipeline.
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
