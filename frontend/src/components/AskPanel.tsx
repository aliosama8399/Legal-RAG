import { useEffect, useRef, useState } from 'react'
import { askStream } from '../lib/api'
import type { Source } from '../lib/types'

interface Props {
  onComplete: (question: string, answer: string) => void
}

export interface LiveEval {
  hit_at_k?: number | null
  reciprocal_rank?: number | null
  faithfulness?: number | null
  answer_relevancy?: number | null
  duration_seconds?: number | null
  degraded?: string | null
}

export default function AskPanel({ onComplete }: Props) {
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(5)
  const [evaluate, setEvaluate] = useState(false)
  const [expectedArticle, setExpectedArticle] = useState('')
  const [liveEval, setLiveEval] = useState<LiveEval | null>(null)
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  async function submit() {
    const trimmed = question.trim()
    if (!trimmed || streaming) return
    setError(null)
    setAnswer('')
    setSources([])
    setLiveEval(null)
    setStreaming(true)
    const controller = new AbortController()
    abortRef.current = controller
    // Local accumulator: React state is stale inside this closure, so the final
    // answer has to be tracked separately to report to history.
    let accumulated = ''
    const expected = expectedArticle.trim() ? Number(expectedArticle) : null
    try {
      await askStream(
        trimmed,
        topK,
        {
          onSources: setSources,
          // Append each token as it arrives — the same progressive output curl shows.
          onToken: (token) => {
            accumulated += token
            setAnswer(accumulated)
          },
          // The terminal event is authoritative. If the stream produced no token
          // events the answer still renders, instead of leaving an empty box.
          onDone: (final) => {
            accumulated = final || accumulated
            setAnswer(accumulated)
          },
          onEval: (scores) => setLiveEval(scores),
        },
        { signal: controller.signal, evaluate, expectedArticle: expected },
      )
      onComplete(trimmed, accumulated)
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setError((err as Error).message)
    } finally {
      setStreaming(false)
    }
  }

  function stop() {
    abortRef.current?.abort()
  }

  return (
    <div className="card">
      <h2>Ask the Egyptian Civil Code</h2>
      <div className="row">
        <div className="grow">
          <label className="small" htmlFor="question">Question</label>
          <textarea
            id="question"
            rows={3}
            value={question}
            placeholder="e.g. What are the grounds for contract nullity?"
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) submit()
            }}
          />
        </div>
      </div>
      <div className="row" style={{ marginTop: 10 }}>
        <div style={{ width: 110 }}>
          <label className="small" htmlFor="topk">top_k</label>
          <input
            id="topk"
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value))}
          />
        </div>
        <div style={{ width: 160 }}>
          <label className="small" htmlFor="expected">gold article (optional)</label>
          <input
            id="expected"
            type="number"
            min={1}
            placeholder="e.g. 450"
            value={expectedArticle}
            onChange={(event) => setExpectedArticle(event.target.value)}
          />
        </div>
        <div style={{ paddingTop: 18 }}>
          {streaming ? (
            <button onClick={stop}>Stop</button>
          ) : (
            <button className="primary" onClick={submit} disabled={!question.trim()}>
              Ask
            </button>
          )}
        </div>
        <span className="badge" style={{ marginTop: 18 }}>
          {streaming ? 'streaming…' : 'ready'}
        </span>
        <label
          className="badge"
          style={{ marginTop: 18, display: 'inline-flex', gap: 6, cursor: 'pointer' }}
        >
          <input
            type="checkbox"
            style={{ width: 'auto' }}
            checked={evaluate}
            onChange={(event) => setEvaluate(event.target.checked)}
          />
          live judge eval {evaluate && '(adds ~9s)'}
        </label>
        <span className="badge" style={{ marginTop: 18 }}>Ctrl+Enter to send</span>
      </div>

      {liveEval && (
        <div className="card" style={{ marginTop: 14, marginBottom: 0 }}>
          <h2>Live evaluation</h2>
          {liveEval.hit_at_k === null && liveEval.hit_at_k === undefined && (
            <p className="hint">
              Retrieval scores need a gold article: enter one above and resend.
              Judge scores need the live judge checkbox.
            </p>
          )}
          <div className="stat">
            <span className="k">hit@k (retrieval)</span>
            <span className={`v ${liveEval.hit_at_k === 1 ? 'good' : 'bad'}`}>
              {liveEval.hit_at_k ?? '—'}
            </span>
          </div>
          <div className="stat">
            <span className="k">reciprocal rank</span>
            <span className="v">{liveEval.reciprocal_rank ?? '—'}</span>
          </div>
          <div className="stat">
            <span className="k">faithfulness (judge)</span>
            <span className={`v ${liveEval.faithfulness === 1 ? 'good' : liveEval.faithfulness === 0 ? 'bad' : ''}`}>
              {liveEval.faithfulness ?? 'not run'}
            </span>
          </div>
          <div className="stat">
            <span className="k">answer relevancy (judge)</span>
            <span className="v">{liveEval.answer_relevancy ?? 'not run'}</span>
          </div>
          <div className="stat">
            <span className="k">live eval latency</span>
            <span className="v">{liveEval.duration_seconds ?? 0}s</span>
          </div>
          <p className="hint">
            These are the same gauges as the Grafana &ldquo;Live evaluation&rdquo;
            panels, persisted so they survive a restart.
          </p>
          {liveEval.degraded && <div className="error">{liveEval.degraded}</div>}
        </div>
      )}

      {error && <div className="error">{error}</div>}

      <div className={`answer${streaming ? ' cursor' : ''}`}>{answer}</div>

      {sources.length > 0 && (
        <div>
          <h2>Sources ({sources.length})</h2>
          {sources.map((source, index) => (
            <div className="source" key={source.chunk_id ?? index}>
              <span className="score">{source.score?.toFixed(4) ?? ''}</span>
              <span className="cite">{source.citation}</span>
              <div className="text">{source.chunk_text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
