import { useEffect, useState } from 'react'
import { history } from '../lib/api'
import type { HistoryItem } from '../lib/types'

interface Props {
  refreshKey: number
  onPick: (question: string) => void
}

export default function HistoryPanel({ refreshKey, onPick }: Props) {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    history(20)
      .then((data) => {
        if (!cancelled) setItems(Array.isArray(data) ? data : [])
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message)
      })
    return () => {
      cancelled = true
    }
  }, [refreshKey])

  return (
    <div className="card">
      <h2>Recent questions</h2>
      {error && <div className="error">{error}</div>}
      {items.length === 0 && !error && <div className="badge">no history yet</div>}
      {items.map((item, index) => (
        <div
          className="history-item"
          key={index}
          onClick={() => onPick(item.question)}
          style={{ cursor: 'pointer' }}
        >
          <div className="q">{item.question}</div>
          <div className="a">{item.answer}</div>
        </div>
      ))}
    </div>
  )
}
