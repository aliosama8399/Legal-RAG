import { useRef, useState } from 'react'
import { embed, upload } from '../lib/api'
import type { EmbedResult, UploadResult } from '../lib/types'

export default function UploadPanel() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploaded, setUploaded] = useState<UploadResult | null>(null)
  const [embedded, setEmbedded] = useState<EmbedResult | null>(null)

  async function run() {
    const file = inputRef.current?.files?.[0]
    if (!file) {
      setError('choose a PDF first')
      return
    }
    setBusy(true)
    setError(null)
    setStatus(`uploading ${file.name} …`)
    setUploaded(null)
    setEmbedded(null)
    try {
      const result = await upload(file)
      setUploaded(result)
      setStatus(`document ${result.document_id}: ${result.articles} articles, ${result.chunks} chunks. embedding …`)
      const embedResult = await embed(result.document_id)
      setEmbedded(embedResult)
      setStatus(
        `embedded ${embedResult.embedded_chunks} chunks (dim ${embedResult.embedding_dimension}) with ${embedResult.embedding_model}`,
      )
    } catch (err) {
      setError((err as Error).message)
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>Ingest a document</h2>
      <label className="small" htmlFor="pdf">PDF (bilingual Egyptian Civil Code)</label>
      <input id="pdf" type="file" accept="application/pdf" ref={inputRef} />
      <div className="row" style={{ marginTop: 10 }}>
        <button onClick={run} disabled={busy}>
          {busy ? 'working…' : 'Upload and embed'}
        </button>
      </div>
      {status && <div className="stat" style={{ marginTop: 12 }}><span className="k">status</span><span className="v">{status}</span></div>}
      {uploaded && (
        <div className="stat"><span className="k">articles</span><span className="v">{uploaded.articles}</span></div>
      )}
      {embedded && (
        <>
          <div className="stat"><span className="k">storage</span><span className="v">{embedded.storage_provider}</span></div>
          <div className="stat"><span className="k">dimension</span><span className="v">{embedded.embedding_dimension}</span></div>
        </>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  )
}
