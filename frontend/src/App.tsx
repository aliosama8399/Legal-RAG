import { useState } from 'react'
import AskPanel from './components/AskPanel'
import HistoryPanel from './components/HistoryPanel'
import UploadPanel from './components/UploadPanel'
import MetricsPanel from './components/MetricsPanel'
import HealthBadge from './components/HealthBadge'

export default function App() {
  // Bumping this re-fetches history after a completed question.
  const [historyKey, setHistoryKey] = useState(0)
  const [pendingQuestion, setPendingQuestion] = useState<string | undefined>(undefined)

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Legal-RAG</h1>
          <div className="sub">Egyptian Civil Code · Qdrant retrieval · vLLM generation</div>
        </div>
        <div className="row">
          <a className="badge" href="http://localhost:3000" target="_blank" rel="noreferrer">
            Langfuse
          </a>
          <a className="badge" href="http://localhost:5000" target="_blank" rel="noreferrer">
            MLflow
          </a>
          <a className="badge" href="http://localhost:9091" target="_blank" rel="noreferrer">
            Grafana
          </a>
          <HealthBadge />
        </div>
      </header>

      <div className="grid">
        <div>
          <AskPanel
            onComplete={() => setHistoryKey((key) => key + 1)}
          />
        </div>
        <div>
          <MetricsPanel />
          <UploadPanel />
          <HistoryPanel refreshKey={historyKey} onPick={setPendingQuestion} />
        </div>
      </div>

      {pendingQuestion && (
        <div className="badge">Selected from history: {pendingQuestion}</div>
      )}
    </div>
  )
}
