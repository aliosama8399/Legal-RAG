import { useEffect, useState } from 'react'
import { metricsSnapshot } from '../lib/api'

interface Props {
  intervalMs?: number
}

export default function MetricsPanel({ intervalMs = 15000 }: Props) {
  const [metrics, setMetrics] = useState<Record<string, number>>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const tick = () => {
      metricsSnapshot()
        .then((data) => {
          if (!cancelled) {
            setMetrics(data)
            setError(null)
          }
        })
        .catch((err: Error) => {
          if (!cancelled) setError(err.message)
        })
    }
    tick()
    const handle = window.setInterval(tick, intervalMs)
    return () => {
      cancelled = true
      window.clearInterval(handle)
    }
  }, [intervalMs])

  // The Prometheus text exposition exposes one series per label combination,
  // so the totals are summed from the raw series rather than read as a single key.
  const total = (predicate: (key: string) => boolean) =>
    Object.entries(metrics)
      .filter(([key]) => predicate(key))
      .reduce((sum, [, value]) => sum + value, 0)

  const requests = total((key) => key.startsWith('rag_requests_total'))
  const promptTokens = total(
    (key) => key.startsWith('rag_llm_tokens_total') && key.includes('prompt'),
  )
  const completionTokens = total(
    (key) => key.startsWith('rag_llm_tokens_total') && key.includes('completion'),
  )
  const pii = total((key) => key.startsWith('rag_pii_redactions_total'))

  const ragas = Object.entries(metrics).filter(([key]) => key.startsWith('rag_ragas_metric{'))
  const metricName = (key: string) => (key.match(/metric="([^"]+)"/)?.[1] ?? key)
  const faithfulness = ragas.find(([key]) => metricName(key) === 'faithfulness')?.[1]
  const questions = metrics.rag_ragas_questions
  // Drift is the DROP against the baseline's own mean. Comparing the raw cosine
  // to 0.90 would flag every reading: baseline questions sit ~0.85 cosine from
  // their own centroid with zero drift.
  const drift = metrics.rag_embedding_drift_cosine
  const driftBaseline = metrics.rag_embedding_drift_baseline_cosine
  const driftDelta = metrics.rag_embedding_drift_delta
  const driftQuestions = metrics.rag_embedding_drift_questions
  const driftDrifting = driftDelta !== undefined && driftDelta < -0.05
  const uptime = metrics.process_start_time_seconds

  return (
    <div className="card">
      <h2>Live metrics</h2>
      {error && <div className="error">{error}</div>}

      <div className="stat">
        <span className="k">Ragas faithfulness</span>
        <span className={`v ${faithfulness === undefined ? '' : faithfulness >= 0.8 ? 'good' : 'bad'}`}>
          {faithfulness === undefined ? 'not evaluated' : faithfulness.toFixed(3)}
        </span>
      </div>
      <div className="stat">
        <span className="k">quality gate</span>
        <span className={`v ${faithfulness === undefined ? '' : faithfulness >= 0.8 ? 'good' : 'bad'}`}>
          {faithfulness === undefined ? '—' : faithfulness >= 0.8 ? 'passing (≥ 0.80)' : 'FAILING (< 0.80)'}
        </span>
      </div>
      <div className="stat">
        <span className="k">questions evaluated</span>
        <span className="v">{questions ?? '—'}</span>
      </div>
      <div className="stat">
        <span className="k">embedding drift</span>
        <span className={`v ${driftQuestions ? (driftDrifting ? 'bad' : 'good') : ''}`}>
          {!driftQuestions
            ? 'no baseline yet'
            : driftDrifting
              ? `drifting (${driftDelta!.toFixed(3)} vs baseline)`
              : `stable (${driftDelta!.toFixed(3)} vs baseline)`}
        </span>
      </div>
      {driftQuestions && drift !== undefined && (
        <div className="stat">
          <span className="k">drift mean / baseline</span>
          <span className="v">
            {drift.toFixed(3)} / {driftBaseline?.toFixed(3) ?? '—'}
          </span>
        </div>
      )}

      {ragas.map(([key, value]) => (
        <div className="stat" key={key}>
          <span className="k">{metricName(key)}</span>
          <span className="v">{value.toFixed(3)}</span>
        </div>
      ))}

      <div className="stat">
        <span className="k">requests (since API start)</span>
        <span className="v">{Math.round(requests).toLocaleString()}</span>
      </div>
      <div className="stat">
        <span className="k">prompt tokens (since API start)</span>
        <span className="v">{Math.round(promptTokens).toLocaleString()}</span>
      </div>
      <div className="stat">
        <span className="k">completion tokens (since API start)</span>
        <span className="v">{Math.round(completionTokens).toLocaleString()}</span>
      </div>
      <div className="stat">
        <span className="k">PII redactions (since API start)</span>
        <span className={`v ${pii > 0 ? 'warn' : ''}`}>{Math.round(pii).toLocaleString()}</span>
      </div>
      {uptime !== undefined && (
        <div className="stat">
          <span className="k">API up since</span>
          <span className="v">{new Date(uptime * 1000).toLocaleString()}</span>
        </div>
      )}

      <div className="badge" style={{ marginTop: 10 }}>
        counters reset when the API container is recreated &middot; rate panels and
        history live in Grafana
      </div>
    </div>
  )
}
