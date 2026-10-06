import { useEffect, useState } from 'react'
import { health } from '../lib/api'

export default function HealthBadge() {
  const [state, setState] = useState<'checking' | 'up' | 'down'>('checking')

  useEffect(() => {
    let cancelled = false
    const tick = () => {
      health()
        .then(() => !cancelled && setState('up'))
        .catch(() => !cancelled && setState('down'))
    }
    tick()
    const handle = window.setInterval(tick, 10000)
    return () => {
      cancelled = true
      window.clearInterval(handle)
    }
  }, [])

  const label = state === 'up' ? 'API healthy' : state === 'down' ? 'API unreachable' : 'checking…'
  const colour = state === 'up' ? 'good' : state === 'down' ? 'bad' : ''

  return (
    <span className={`badge ${colour}`} style={{ color: colour === 'good' ? 'var(--good)' : colour === 'bad' ? 'var(--bad)' : undefined }}>
      {label}
    </span>
  )
}
