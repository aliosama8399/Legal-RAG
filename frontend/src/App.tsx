import { useEffect, useState } from 'react'
import AskPanel from './components/AskPanel'
import HealthBadge from './components/HealthBadge'
import MetricsPanel from './components/MetricsPanel'
import UploadPanel from './components/UploadPanel'
import {
  clearAuth,
  getMe,
  loginUser,
  registerOrg,
  setApiKeyHeader,
  setAuthToken,
} from './lib/api'
import type { UserProfile } from './lib/types'

type NavTab = 'copilot' | 'knowledge' | 'admin' | 'telemetry'

export default function App() {
  const [tab, setTab] = useState<NavTab>('copilot')
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [authModalOpen, setAuthModalOpen] = useState(false)
  const [authMode, setAuthMode] = useState<'login' | 'register' | 'apikey'>('login')
  const [orgName, setOrgName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [authError, setAuthError] = useState<string | null>(null)
  const [refreshTick, setRefreshTick] = useState(0)

  async function loadProfile() {
    try {
      const me = await getMe()
      setProfile(me)
    } catch {
      setProfile(null)
    }
  }

  useEffect(() => {
    loadProfile()
  }, [refreshTick])

  const tenantKey = `${profile?.tenant_id ?? 'anon'}:${refreshTick}`

  async function handleAuthSubmit() {
    setAuthError(null)
    try {
      if (authMode === 'register') {
        const res = await registerOrg({ org_name: orgName, email, password })
        setAuthToken(res.access_token)
        setProfile(res.profile)
      } else if (authMode === 'login') {
        const res = await loginUser(email, password)
        setAuthToken(res.access_token)
        setProfile(res.profile)
      } else {
        setApiKeyHeader(apiKeyInput.trim())
        const me = await getMe()
        setProfile(me)
      }
      setAuthModalOpen(false)
      setRefreshTick((t) => t + 1)
    } catch (err) {
      setAuthError((err as Error).message)
    }
  }

  function handleLogout() {
    clearAuth()
    setRefreshTick((t) => t + 1)
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Legal-RAG Enterprise</h1>
          <div className="sub">
            Multi-Tenant Legal AI Workspace · Hybrid Retrieval · RBAC · FinOps Metering
          </div>
        </div>
        <div className="row">
          {profile ? (
            <>
              <span className="badge good">
                {profile.org_name} · {profile.role} ({profile.plan})
              </span>
              <button onClick={() => setAuthModalOpen(true)} style={{ padding: '4px 10px', fontSize: 12 }}>
                Switch Workspace
              </button>
              {profile.auth_type !== 'anonymous' && (
                <button onClick={handleLogout} style={{ padding: '4px 10px', fontSize: 12 }}>
                  Sign Out
                </button>
              )}
            </>
          ) : (
            <button className="primary" onClick={() => setAuthModalOpen(true)} style={{ padding: '5px 12px', fontSize: 12 }}>
              Sign In / Register Org
            </button>
          )}
          <a className="badge" href="http://localhost:3000" target="_blank" rel="noreferrer">Langfuse</a>
          <a className="badge" href="http://localhost:5000" target="_blank" rel="noreferrer">MLflow</a>
          <a className="badge" href="http://localhost:9091" target="_blank" rel="noreferrer">Grafana</a>
          <HealthBadge />
        </div>
      </header>

      <nav className="nav-tabs">
        <button className={tab === 'copilot' ? 'primary' : ''} onClick={() => setTab('copilot')}>
          Research Copilot
        </button>
        <button className={tab === 'knowledge' ? 'primary' : ''} onClick={() => setTab('knowledge')}>
          Knowledge Base
        </button>
        <button className={tab === 'admin' ? 'primary' : ''} onClick={() => setTab('admin')}>
          Tenant Admin &amp; FinOps
        </button>
        <button className={tab === 'telemetry' ? 'primary' : ''} onClick={() => setTab('telemetry')}>
          System Telemetry
        </button>
      </nav>

      {tab === 'copilot' && (
        <AskPanel tenantKey={tenantKey} onComplete={() => setRefreshTick((t) => t + 1)} />
      )}
      {tab === 'knowledge' && <UploadPanel tenantKey={tenantKey} />}
      {tab === 'admin' && <MetricsPanel tenantKey={tenantKey} mode="admin" />}
      {tab === 'telemetry' && <MetricsPanel tenantKey={tenantKey} mode="telemetry" />}

      {authModalOpen && (
        <div className="modal-backdrop" onClick={() => setAuthModalOpen(false)}>
          <div className="card modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="row" style={{ justifyContent: 'space-between', marginBottom: 14 }}>
              <h2 style={{ margin: 0 }}>Organization Authentication</h2>
              <button onClick={() => setAuthModalOpen(false)}>×</button>
            </div>
            <div className="row" style={{ marginBottom: 14 }}>
              <button className={authMode === 'login' ? 'primary' : ''} onClick={() => setAuthMode('login')}>
                Sign In
              </button>
              <button className={authMode === 'register' ? 'primary' : ''} onClick={() => setAuthMode('register')}>
                Register Org
              </button>
              <button className={authMode === 'apikey' ? 'primary' : ''} onClick={() => setAuthMode('apikey')}>
                API Key
              </button>
            </div>

            {authMode === 'register' && (
              <div style={{ marginBottom: 10 }}>
                <label className="small">Organization Name</label>
                <input value={orgName} onChange={(e) => setOrgName(e.target.value)} placeholder="Acme Legal LLP" />
              </div>
            )}

            {authMode !== 'apikey' ? (
              <>
                <div style={{ marginBottom: 10 }}>
                  <label className="small">Email</label>
                  <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="partner@firm.com" />
                </div>
                <div style={{ marginBottom: 12 }}>
                  <label className="small">Password</label>
                  <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
                </div>
              </>
            ) : (
              <div style={{ marginBottom: 12 }}>
                <label className="small">Scoped API Key (lrag_live_...)</label>
                <input value={apiKeyInput} onChange={(e) => setApiKeyInput(e.target.value)} placeholder="lrag_live_..." />
              </div>
            )}

            {authError && <div className="error" style={{ marginBottom: 12 }}>{authError}</div>}
            <button className="primary" onClick={handleAuthSubmit} style={{ width: '100%' }}>
              Authenticate Workspace
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
