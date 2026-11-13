// The AI panel (always on the right): shows live MCP agent activity, concepts, and undo.
import { useEffect, useRef } from 'react'
import type { Activity, Concept } from '../api'
import { Icon } from '../ui/icons'

interface Props {
  activity: Activity[]
  context: string | null
  concept: Concept | null // the newest
  collapsed?: boolean
  onToggleCollapse?: () => void
  onOpenConcept: (id: string) => void
  onRevert: () => void
}

const time = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

export function AiPanel({ activity, context, concept, collapsed = false, onToggleCollapse, onOpenConcept, onRevert }: Props) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => { end.current?.scrollIntoView({ block: 'end' }) }, [activity.length])
  // The AI's changes, and the user's answers to its concepts (the conversation about what to build).
  const ai = activity.filter((a) => a.actor !== 'user' || a.text.startsWith('Concept '))

  return (
    <aside className={`side ai${collapsed ? ' collapsed' : ''}`} aria-label="AI assistant">
      {collapsed ? (
        <button type="button" className="side-rail" onClick={onToggleCollapse} aria-expanded={false}
          aria-label="Expand AI assistant panel" title="Expand AI assistant panel">
          <Icon name="chevL" size={14} />
          <span className="side-rail-label">AI assistant</span>
        </button>
      ) : null}
      <div className="side-body" hidden={collapsed}>
        <div className="aihead">
          {onToggleCollapse && (
            <button type="button" className="side-toggle" onClick={onToggleCollapse} aria-expanded={true}
              aria-label="Collapse AI assistant panel" title="Collapse AI assistant panel">
              <Icon name="chevR" size={14} />
            </button>
          )}
          <h2>AI assistant</h2>
          <span className="spacer" />
          <span className="chip">MCP</span>
        </div>
        {context && <div className="ctx">Context: {context}</div>}
        {concept && (
          <div className="ccard">
            <div className="row">
              <b>{concept.id} · {concept.title}</b><span className="spacer" />
              <span className={`chip cstat ${concept.status}`}>{concept.status === 'pending' ? 'Waiting for you' : concept.status === 'approved' ? 'Approved' : 'Changes asked'}</span>
            </div>
            <img src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(concept.svg)}`} alt={`Concept drawing: ${concept.title}`}
              onClick={() => onOpenConcept(concept.id)} />
            <button className="btn sm" onClick={() => onOpenConcept(concept.id)}>{concept.status === 'pending' ? 'Answer' : 'Open'}</button>
          </div>
        )}

        <div className="log">
          {ai.length === 0 && (
            <div className="intro">
              <p>Connect Antigravity, Claude Code, or any MCP client while this app is open. It drives the
                <code> 3d-print-assist</code> MCP server with the same tools you have; you watch each change land here
                and in the tree, and can edit alongside it.</p>
            </div>
          )}
          {ai.map((a, i) => (
            <div key={i} className={`entry ${a.ok ? 'tool' : 'error'}`}>
              <span className="who">{a.actor}</span><span className="when">{time(a.at)}</span>
              <div>{a.text}</div>
            </div>
          ))}
          <div ref={end} />
        </div>
        <div className="aifoot">
          <button className="btn" onClick={onRevert} disabled={!ai.some((a) => a.ok && a.actor !== 'user')}>Revert last AI change</button>
        </div>
      </div>
    </aside>
  )
}
