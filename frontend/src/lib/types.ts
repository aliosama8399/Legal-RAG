export interface Source {
  chunk_id?: string
  chunk_text: string
  citation: string
  score?: number
  article_number?: number | string
  document_id?: number
}

export interface AskResponse {
  question: string
  answer: string
  sources: Source[]
  llm_model: string
  embedding_model: string
  storage_provider: string
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
  | { type: 'done'; answer: string; sources: Source[] }
  | ({ type: 'eval' } & LiveEvalScores)
  | { type: 'error'; error: string }

export interface HistoryItem {
  id?: number
  document_id?: number | null
  question: string
  answer: string
  created_at?: string
  sources?: Source[]
}

/** The API wraps history in `results`; it is not a bare array. */
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
