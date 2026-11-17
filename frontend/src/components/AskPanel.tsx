import { useEffect, useRef, useState } from 'react'
import {
  askStream,
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  listDocuments,
  sendConversationMessage,
  submitFeedback,
} from '../lib/api'
import type {
  ConversationDetail,
  ConversationSummary,
  DocumentSummary,
  LiveEvalScores,
  Source,
} from '../lib/types'

interface Props {
  onComplete: () => void
  tenantKey: string
}

export default function AskPanel({ onComplete, tenantKey }: Props) {
  const [mode, setMode] = useState<'thread' | 'stream'>('thread')
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeConvId, setActiveConvId] = useState<string | null>(null)
  const [activeConv, setActiveConv] = useState<ConversationDetail | null>(null)
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null)

  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(5)
  const [evaluate, setEvaluate] = useState(false)
  const [expectedArticle, setExpectedArticle] = useState('')
  const [liveEval, setLiveEval] = useState<LiveEvalScores | null>(null)

  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [selectedSource, setSelectedSource] = useState<Source | null>(null)
  const [lastTraceId, setLastTraceId] = useState<string>('')
  const [feedbackSent, setFeedbackSent] = useState<Record<string, 0 | 1>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  async function refreshWorkspace() {
    try {
      const [convs, docs] = await Promise.all([listConversations(), listDocuments()])
      setConversations(convs)
      setDocuments(docs)
    } catch {
      // Unauthenticated when auth_required=true: parent shows login prompt
    }
  }

  useEffect(() => {
    setActiveConvId(null)
    setActiveConv(null)
    setAnswer('')
    setSources([])
    setSelectedSource(null)
    refreshWorkspace()
  }, [tenantKey])

  async function selectConversation(convId: string) {
    setError(null)
    setActiveConvId(convId)
    try {
      const detail = await getConversation(convId)
      setActiveConv(detail)
      const lastAssistant = [...detail.messages].reverse().find((m) => m.role === 'assistant')
      setSources(lastAssistant?.sources ?? [])
      setSelectedSource(lastAssistant?.sources?.[0] ?? null)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function handleNewThread() {
    setError(null)
    try {
      const created = await createConversation('Legal Research Thread', selectedDocId)
      await refreshWorkspace()
      await selectConversation(created.id)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function handleDeleteThread(convId: string) {
    try {
      await deleteConversation(convId)
      if (activeConvId === convId) {
        setActiveConvId(null)
        setActiveConv(null)
      }
      await refreshWorkspace()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function sendThreadTurn() {
    const trimmed = question.trim()
    if (!trimmed || busy) return
    setBusy(true)
    setError(null)
    setLiveEval(null)
    const expected = expectedArticle.trim() ? Number(expectedArticle) : null
    try {
      let convId = activeConvId
      if (!convId) {
        const created = await createConversation(trimmed.slice(0, 60), selectedDocId)
        convId = created.id
        setActiveConvId(convId)
      }
      const res = await sendConversationMessage(convId, trimmed, topK, selectedDocId, {
        evaluate,
        expectedArticle: expected,
      })
      setQuestion('')
      setSources(res.sources)
      setSelectedSource(res.sources[0] ?? null)
      setLastTraceId(res.assistant_message.trace_id)
      if (res.live_eval) setLiveEval(res.live_eval)
      const detail = await getConversation(convId)
      setActiveConv(detail)
      await refreshWorkspace()
      onComplete()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function sendStreamTurn() {
    const trimmed = question.trim()
    if (!trimmed || busy) return
    setError(null)
    setAnswer('')
    setSources([])
    setSelectedSource(null)
    setLiveEval(null)
    setLastTraceId('')
    setBusy(true)
    const controller = new AbortController()
    abortRef.current = controller
    let accumulated = ''
    const expected = expectedArticle.trim() ? Number(expectedArticle) : null
    try {
      await askStream(
        trimmed,
        topK,
        {
          onSources: (hits) => {
            setSources(hits)
            setSelectedSource(hits[0] ?? null)
          },
          onToken: (token) => {
            accumulated += token
            setAnswer(accumulated)
          },
          onDone: (final, traceId) => {
            accumulated = final || accumulated
            setAnswer(accumulated)
            if (traceId) setLastTraceId(traceId)
          },
          onEval: (scores) => setLiveEval(scores),
        },
        {
          signal: controller.signal,
          evaluate,
          expectedArticle: expected,
          documentId: selectedDocId,
        },
      )
      onComplete()
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function handleFeedback(traceId: string, score: 0 | 1, messageId = '') {
    if (!traceId) return
    try {
      await submitFeedback(traceId, score, messageId)
      setFeedbackSent((prev) => ({ ...prev, [traceId]: score }))
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div className="copilot-layout">
      <aside className="card sidebar-card">
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Research Threads</h2>
          <button className="primary" onClick={handleNewThread} style={{ padding: '4px 10px', fontSize: 12 }}>
            + New
          </button>
        </div>

        <label className="small" htmlFor="doc-scope">Document Scope</label>
        <select
          id="doc-scope"
          value={selectedDocId ?? ''}
          onChange={(e) => setSelectedDocId(e.target.value ? Number(e.target.value) : null)}
          style={{ marginBottom: 12 }}
        >
          <option value="">All Tenant Documents</option>
          {documents.map((doc) => (
            <option key={doc.document_id} value={doc.document_id}>
              #{doc.document_id} · {doc.filename} ({doc.status})
            </option>
          ))}
        </select>

        <div className="thread-list">
          {conversations.length === 0 && (
            <div className="small" style={{ color: 'var(--muted)' }}>
              No threads yet. Ask a question to start one.
            </div>
          )}
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`thread-item ${activeConvId === conv.id ? 'active' : ''}`}
              onClick={() => {
                setMode('thread')
                selectConversation(conv.id)
              }}
            >
              <div className="thread-title">{conv.title}</div>
              <div className="row" style={{ justifyContent: 'space-between', marginTop: 4 }}>
                <span className="badge">{conv.message_count} msgs</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDeleteThread(conv.id)
                  }}
                  style={{ padding: '2px 6px', fontSize: 11 }}
                >
                  ×
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      <div className="copilot-main">
        <div className="card">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
            <h2 style={{ margin: 0 }}>Legal Research Copilot</h2>
            <div className="row">
              <button
                className={mode === 'thread' ? 'primary' : ''}
                onClick={() => setMode('thread')}
                style={{ fontSize: 12, padding: '5px 10px' }}
              >
                Multi-Turn Thread
              </button>
              <button
                className={mode === 'stream' ? 'primary' : ''}
                onClick={() => setMode('stream')}
                style={{ fontSize: 12, padding: '5px 10px' }}
              >
                Live SSE Stream
              </button>
            </div>
          </div>

          {mode === 'thread' && activeConv && (
            <div className="chat-timeline">
              {activeConv.messages.map((msg) => (
                <div key={msg.id} className={`chat-bubble ${msg.role}`}>
                  <div className="chat-role">
                    {msg.role === 'user' ? 'Counsel' : 'Legal-RAG Copilot'}
                  </div>
                  <div className="chat-content">{msg.content}</div>
                  {msg.role === 'assistant' && msg.sources.length > 0 && (
                    <div className="row" style={{ marginTop: 8, gap: 6 }}>
                      {msg.sources.map((src, idx) => (
                        <button
                          key={src.chunk_id ?? idx}
                          className="cite-pill"
                          onClick={() => setSelectedSource(src)}
                        >
                          {src.citation}
                        </button>
                      ))}
                      {msg.trace_id && (
                        <div className="row" style={{ marginLeft: 'auto', gap: 4 }}>
                          <button
                            onClick={() => handleFeedback(msg.trace_id, 1, msg.id)}
                            className={feedbackSent[msg.trace_id] === 1 ? 'primary' : ''}
                            style={{ padding: '2px 8px', fontSize: 12 }}
                          >
                            👍
                          </button>
                          <button
                            onClick={() => handleFeedback(msg.trace_id, 0, msg.id)}
                            className={feedbackSent[msg.trace_id] === 0 ? 'primary' : ''}
                            style={{ padding: '2px 8px', fontSize: 12 }}
                          >
                            👎
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {mode === 'stream' && answer && (
            <div className={`answer${busy ? ' cursor' : ''}`}>
              {answer}
              {lastTraceId && !busy && (
                <div className="row" style={{ marginTop: 8 }}>
                  <button
                    onClick={() => handleFeedback(lastTraceId, 1)}
                    className={feedbackSent[lastTraceId] === 1 ? 'primary' : ''}
                    style={{ padding: '2px 8px', fontSize: 12 }}
                  >
                    👍 Helpful
                  </button>
                  <button
                    onClick={() => handleFeedback(lastTraceId, 0)}
                    className={feedbackSent[lastTraceId] === 0 ? 'primary' : ''}
                    style={{ padding: '2px 8px', fontSize: 12 }}
                  >
                    👎 Needs improvement
                  </button>
                </div>
              )}
            </div>
          )}

          <label className="small" htmlFor="question">Ask a legal question or follow-up</label>
          <textarea
            id="question"
            rows={3}
            value={question}
            placeholder="e.g. What does Article 147 state regarding contract binding force?"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                mode === 'thread' ? sendThreadTurn() : sendStreamTurn()
              }
            }}
          />

          <div className="row" style={{ marginTop: 10 }}>
            <div style={{ width: 90 }}>
              <label className="small" htmlFor="topk">top_k</label>
              <input
                id="topk"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>

            <div style={{ width: 150 }}>
              <label className="small" htmlFor="expected">gold article (auto if empty)</label>
              <input
                id="expected"
                type="number"
                min={1}
                placeholder="auto / e.g. 147"
                value={expectedArticle}
                onChange={(e) => setExpectedArticle(e.target.value)}
              />
            </div>
            <label className="badge" style={{ marginTop: 18, display: 'inline-flex', gap: 6, cursor: 'pointer' }}>
              <input
                type="checkbox"
                style={{ width: 'auto' }}
                checked={evaluate}
                onChange={(e) => setEvaluate(e.target.checked)}
              />
              live judge
            </label>

            <div style={{ paddingTop: 18, marginLeft: 'auto' }}>
              {busy && mode === 'stream' ? (
                <button onClick={() => abortRef.current?.abort()}>Stop</button>
              ) : (
                <button
                  className="primary"
                  disabled={busy || !question.trim()}
                  onClick={() => (mode === 'thread' ? sendThreadTurn() : sendStreamTurn())}
                >
                  {busy ? 'Analyzing…' : mode === 'thread' ? 'Send Message' : 'Stream Answer'}
                </button>
              )}
            </div>
          </div>

          {liveEval && (
            <div className="card" style={{ marginTop: 14, marginBottom: 0 }}>
              <h2>Live Evaluation</h2>
              <div className="stat"><span className="k">hit@k</span><span className="v">{liveEval.hit_at_k !== null ? liveEval.hit_at_k.toFixed(3) : '—'}</span></div>
              <div className="stat"><span className="k">reciprocal rank</span><span className="v">{liveEval.reciprocal_rank !== null ? liveEval.reciprocal_rank.toFixed(3) : '—'}</span></div>
              <div className="stat"><span className="k">faithfulness</span><span className="v">{liveEval.faithfulness !== null ? liveEval.faithfulness.toFixed(3) : '—'}</span></div>
              <div className="stat"><span className="k">answer relevancy</span><span className="v">{liveEval.answer_relevancy !== null ? liveEval.answer_relevancy.toFixed(3) : '—'}</span></div>
              {liveEval.duration_seconds !== null && (
                <div className="stat"><span className="k">eval latency</span><span className="v">{liveEval.duration_seconds.toFixed(2)}s</span></div>
              )}
            </div>
          )}

          {error && <div className="error">{error}</div>}
        </div>

        {sources.length > 0 && (
          <div className="card">
            <h2>Retrieved Citations & Bilingual Source Inspector ({sources.length})</h2>
            <div className="grid-2">
              <div>
                {sources.map((src, idx) => (
                  <div
                    key={src.chunk_id ?? idx}
                    className={`source ${selectedSource?.chunk_id === src.chunk_id ? 'selected' : ''}`}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSelectedSource(src)}
                  >
                    <span className="score">{src.score?.toFixed(4) ?? ''}</span>
                    <span className="cite">{src.citation}</span>
                    <div className="text">{src.chunk_text.slice(0, 140)}…</div>
                  </div>
                ))}
              </div>
              {selectedSource && (
                <div className="source-inspector">
                  <div className="cite" style={{ marginBottom: 8 }}>{selectedSource.citation}</div>
                  {selectedSource.text_ar && (
                    <div dir="rtl" className="arabic-box">{selectedSource.text_ar}</div>
                  )}
                  <div className="text" style={{ color: 'var(--text)' }}>
                    {selectedSource.text_en || selectedSource.chunk_text}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
