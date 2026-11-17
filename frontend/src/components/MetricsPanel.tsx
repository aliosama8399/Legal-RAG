import { useEffect, useState } from 'react'
import {
  createApiKey,
  getAuditLogs,
  getTenantUsage,
  listApiKeys,
  metricsSnapshot,
  revokeApiKey,
} from '../lib/api'
import type { ApiKeyItem, AuditLogEntry, UsageSummary } from '../lib/types'

interface Props {
  intervalMs?: number
  tenantKey: string
  mode?: 'admin' | 'telemetry'
}

export default function MetricsPanel({ intervalMs = 15000, tenantKey, mode = 'admin' }: Props) {
  const [metrics, setMetrics] = useState<Record<string, number>>({})
  const [usage, setUsage] = useState<UsageSummary | null>(null)
  const [apiKeys, setApiKeys] = useState<ApiKeyItem[]>([])
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([])
  const [newKeyName, setNewKeyName] = useState('')
  const [revealedKey, setRevealedKey] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function refreshAdmin() {
    try {
      const [u, keys, logs] = await Promise.all([
        getTenantUsage(),
        listApiKeys(),
        getAuditLogs(30).catch(() => []),
      ])
      setUsage(u)
      setApiKeys(keys)
      setAuditLogs(logs)
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  useEffect(() => {
    let cancelled = false
    if (mode === 'admin') {
      refreshAdmin()
      return
    }
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
  }, [intervalMs, tenantKey, mode])

  async function handleCreateKey() {
    if (!newKeyName.trim()) return
    setError(null)
    try {
      const created = await createApiKey(newKeyName.trim())
      setRevealedKey(created.api_key)
      setNewKeyName('')
      await refreshAdmin()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function handleRevoke(keyId: string) {
    try {
      await revokeApiKey(keyId)
      await refreshAdmin()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  if (mode === 'admin') {
    return (
      <div className="grid-2">
        <div className="card">
          <h2>Organization Metering & Quotas</h2>
          {error && <div className="error">{error}</div>}
          {usage && (
            <>
              <div className="stat"><span className="k">Organization</span><span className="v">{usage.org_name} ({usage.plan})</span></div>
              <div className="stat"><span className="k">API Operations</span><span className="v">{usage.total_operations}</span></div>
              <div className="stat"><span className="k">Total Tokens</span><span className="v">{usage.total_tokens.toLocaleString()} / {usage.monthly_token_quota.toLocaleString()}</span></div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${Math.min(100, usage.quota_utilization_pct)}%` }} />
              </div>
              <div className="stat"><span className="k">Quota Utilization</span><span className="v">{usage.quota_utilization_pct}%</span></div>
              <div className="stat"><span className="k">Estimated Cost (USD)</span><span className="v good">${usage.estimated_cost_usd.toFixed(6)}</span></div>
              <div className="stat"><span className="k">Documents</span><span className="v">{usage.documents_count} / {usage.max_documents}</span></div>
            </>
          )}

          <h2 style={{ marginTop: 20 }}>Scoped API Keys</h2>
          <div className="row" style={{ marginBottom: 10 }}>
            <div className="grow">
              <input
                placeholder="Key name (e.g. prod-webhook)"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
              />
            </div>
            <button className="primary" onClick={handleCreateKey} disabled={!newKeyName.trim()}>
              Mint Key
            </button>
          </div>
          {revealedKey && (
            <div className="card" style={{ background: 'var(--panel-2)', marginBottom: 12 }}>
              <div className="small" style={{ color: 'var(--good)' }}>Copy your secret API key now (shown once):</div>
              <code style={{ wordBreak: 'break-all', fontSize: 12 }}>{revealedKey}</code>
            </div>
          )}
          {apiKeys.map((k) => (
            <div key={k.id} className="stat">
              <span>
                <strong>{k.name}</strong> <span className="badge">{k.key_prefix}</span>
              </span>
              <span className="row">
                <span className={`badge ${k.active ? 'good' : 'bad'}`}>{k.active ? 'active' : 'revoked'}</span>
                {k.active && (
                  <button onClick={() => handleRevoke(k.id)} style={{ padding: '2px 8px', fontSize: 11 }}>
                    Revoke
                  </button>
                )}
              </span>
            </div>
          ))}
        </div>

        <div className="card">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
            <h2 style={{ margin: 0 }}>SOC2 Audit Log ({auditLogs.length})</h2>
            <button onClick={refreshAdmin} style={{ padding: '4px 10px', fontSize: 12 }}>Refresh</button>
          </div>
          {auditLogs.length === 0 ? (
            <div className="badge">No audit entries or admin role required</div>
          ) : (
            <div style={{ maxHeight: 420, overflowY: 'auto' }}>
              {auditLogs.map((log) => (
                <div key={log.id} className="history-item">
                  <div className="row" style={{ justifyContent: 'space-between' }}>
                    <span className="q">{log.action}</span>
                    <span className="small" style={{ color: 'var(--muted)' }}>
                      {new Date(log.created_at).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="a">
                    {log.resource_type}: {log.resource_id} · actor: {log.actor_id}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

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
  const driftDelta = metrics.rag_embedding_drift_delta
  const driftQuestions = metrics.rag_embedding_drift_questions

  return (
    <div className="card">
      <h2>Prometheus & Quality Gate Telemetry</h2>
      {error && <div className="error">{error}</div>}
      <div className="stat">
        <span className="k">Ragas faithfulness</span>
        <span className={`v ${faithfulness === undefined ? '' : faithfulness >= 0.8 ? 'good' : 'bad'}`}>
          {faithfulness === undefined ? 'not evaluated' : faithfulness.toFixed(3)}
        </span>
      </div>
      <div className="stat">
        <span className="k">Embedding drift</span>
        <span className="v">{!driftQuestions ? 'no baseline yet' : `${driftDelta?.toFixed(3)}`}</span>
      </div>
      <div className="stat"><span className="k">Requests</span><span className="v">{Math.round(requests).toLocaleString()}</span></div>
      <div className="stat"><span className="k">Prompt tokens</span><span className="v">{Math.round(promptTokens).toLocaleString()}</span></div>
      <div className="stat"><span className="k">Completion tokens</span><span className="v">{Math.round(completionTokens).toLocaleString()}</span></div>
      <div className="stat"><span className="k">PII redactions</span><span className="v">{Math.round(pii).toLocaleString()}</span></div>
    </div>
  )
}
