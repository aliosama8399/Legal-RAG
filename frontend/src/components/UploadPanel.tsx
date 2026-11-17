import { useEffect, useRef, useState } from 'react'
import {
  deleteDocument,
  getIngestionJob,
  ingestDocumentAsync,
  listDocuments,
} from '../lib/api'
import type { DocumentSummary, IngestionJob } from '../lib/types'

interface Props {
  tenantKey: string
}

export default function UploadPanel({ tenantKey }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [job, setJob] = useState<IngestionJob | null>(null)
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [error, setError] = useState<string | null>(null)

  async function refreshDocs() {
    try {
      const docs = await listDocuments()
      setDocuments(docs)
    } catch {
      // ignore if unauthenticated
    }
  }

  useEffect(() => {
    refreshDocs()
  }, [tenantKey])

  async function runAsyncIngest() {
    const file = inputRef.current?.files?.[0]
    if (!file) {
      setError('Choose a PDF file first.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const idempKey = `${file.name}:${file.size}:${file.lastModified}`
      let current = await ingestDocumentAsync(file, idempKey)
      setJob(current)

      while (current.status !== 'READY' && current.status !== 'FAILED') {
        await new Promise((r) => setTimeout(r, 800))
        current = await getIngestionJob(current.id)
        setJob(current)
      }
      if (current.status === 'FAILED') {
        setError(current.error || 'Async ingestion failed.')
      } else {
        await refreshDocs()
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function handleDelete(documentId: number) {
    setError(null)
    try {
      await deleteDocument(documentId)
      await refreshDocs()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div className="grid-2">
      <div className="card">
        <h2>Asynchronous Ingestion Pipeline (HTTP 202)</h2>
        <label className="small" htmlFor="pdf">Upload Legal PDF Corpus</label>
        <input id="pdf" type="file" accept="application/pdf" ref={inputRef} />
        <div className="row" style={{ marginTop: 12 }}>
          <button className="primary" onClick={runAsyncIngest} disabled={busy}>
            {busy ? 'Processing Job…' : 'Enqueue Async Ingest'}
          </button>
        </div>

        {job && (
          <div style={{ marginTop: 16 }}>
            <div className="stat"><span className="k">Job ID</span><span className="v">{job.id}</span></div>
            <div className="stat">
              <span className="k">Status / Stage</span>
              <span className={`v ${job.status === 'READY' ? 'good' : job.status === 'FAILED' ? 'bad' : ''}`}>
                {job.status} ({job.stage})
              </span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${job.progress_pct}%` }} />
            </div>
            <div className="stat"><span className="k">Progress</span><span className="v">{job.progress_pct}%</span></div>
            {job.chunks > 0 && (
              <div className="stat"><span className="k">Chunks Indexed</span><span className="v">{job.chunks}</span></div>
            )}
          </div>
        )}
        {error && <div className="error">{error}</div>}
      </div>

      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Tenant Knowledge Base ({documents.length})</h2>
          <button onClick={refreshDocs} style={{ padding: '4px 10px', fontSize: 12 }}>Refresh</button>
        </div>
        {documents.length === 0 ? (
          <div className="badge">No documents indexed for this organization yet</div>
        ) : (
          <div className="table-wrap">
            {documents.map((doc) => (
              <div key={doc.document_id} className="doc-row">
                <div>
                  <div style={{ fontWeight: 600 }}>#{doc.document_id} · {doc.filename}</div>
                  <div className="small" style={{ color: 'var(--muted)' }}>
                    {doc.article_count} articles · {doc.chunk_count} chunks · {doc.embedding_model || 'pending'}
                  </div>
                </div>
                <div className="row">
                  <span className={`badge ${doc.status === 'ready' ? 'good' : ''}`}>{doc.status}</span>
                  <button
                    onClick={() => handleDelete(doc.document_id)}
                    style={{ padding: '4px 8px', fontSize: 12 }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
