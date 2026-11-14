// A concept the AI proposes before it builds: its drawing, what it will make, what it assumed and what it asks.
// Shown large over the 3D view while it waits; the user approves it or says what to change.
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import type { Concept } from '../api'

const svgUrl = (svg: string) => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
const STATUS = { pending: 'Waiting for you', approved: 'Approved', revise: 'Changes asked' } as const

interface Props {
  concept: Concept
  hideAnswerInput?: boolean
  onAnswer: (decision: 'approve' | 'revise', note: string) => void
  onBuild?: (note: string) => void
  onClose: () => void
}

export function ConceptView({ concept: c, hideAnswerInput, onAnswer, onBuild, onClose }: Props) {
  const [note, setNote] = useState('')
  const [expanded, setExpanded] = useState(false)
  const [focusPane, setFocusPane] = useState<'both' | 'draw' | 'spec'>('both')
  const pending = c.status === 'pending'

  useEffect(() => {
    if (!expanded) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopPropagation()
        setExpanded(false)
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [expanded])

  // Full screen covers the entire app UI (portaled to document.body above all panels); Esc, button, or double-click exits.
  const fullScreen = () => setExpanded((v) => !v)
  return (
    <section className="concept" role="dialog" aria-label={`Concept ${c.id}: ${c.title}`}>
      <div className="dhead">
        <h2>{c.id} · {c.title}</h2>
        <span className={`chip cstat ${c.status}`}>{STATUS[c.status]}</span>
        <span className="spacer" />
        <button className="btn" onClick={fullScreen} title="Show the drawing full screen (Esc to come back; or double-click the drawing)">⛶ Full screen</button>
        <button className="x" onClick={onClose} aria-label={pending ? 'Hide for now' : 'Close'} title={pending ? 'Hide for now (it stays in the AI panel)' : 'Close'}>×</button>
      </div>
      <div className="cbody">
        <div className={`cpane cdraw-pane${focusPane === 'draw' ? ' is-full' : focusPane === 'spec' ? ' is-collapsed' : ''}`}>
          <div className="cpane-bar">
            <span className="cpane-title">Drawing</span>
            <span className="spacer" />
            <button type="button" className="btn sm" onClick={() => setFocusPane(focusPane === 'draw' ? 'both' : 'draw')}>
              {focusPane === 'draw' ? 'Restore' : 'Expand'}
            </button>
          </div>
          {focusPane !== 'spec' && (
            <div className="cdraw" onDoubleClick={fullScreen} title="Double-click for full screen">
              <img src={svgUrl(c.svg)} alt={`Concept drawing: ${c.title}`} />
            </div>
          )}
        </div>
        {expanded && createPortal(
          <div className="cdraw-fullscreen" role="dialog" aria-label={`Full screen drawing: ${c.title}`} onDoubleClick={() => setExpanded(false)}>
            <button type="button" className="btn cdraw-exit" onClick={() => setExpanded(false)} title="Exit full screen (Esc)">
              ✕ Exit full screen
            </button>
            <img src={svgUrl(c.svg)} alt={`Concept drawing: ${c.title}`} />
          </div>,
          document.body,
        )}
        <div className={`cpane ctext-pane${focusPane === 'spec' ? ' is-full' : focusPane === 'draw' ? ' is-collapsed' : ''}`}>
          <div className="cpane-bar">
            <span className="cpane-title">Spec, Assumptions &amp; Questions</span>
            <span className="spacer" />
            <button type="button" className="btn sm" onClick={() => setFocusPane(focusPane === 'spec' ? 'both' : 'spec')}>
              {focusPane === 'spec' ? 'Restore' : 'Expand'}
            </button>
          </div>
          {focusPane !== 'draw' && (
            <div className="ctext">
              {c.summary && <p className="csum">{c.summary}</p>}
              {(c.assumptions.length > 0 || c.questions.length > 0) && (
                <div className="ctext-cols">
                  {c.assumptions.length > 0 && (<div><h3>Assumed</h3><ul>{c.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul></div>)}
                  {c.questions.length > 0 && (<div><h3>Questions</h3><ul className="qs">{c.questions.map((q, i) => <li key={i}>{q}</li>)}</ul></div>)}
                </div>
              )}
          {!hideAnswerInput && (pending ? (
            <div className="canswer">
              <label htmlFor="c-note" className="nm">Your answer (answers to the questions, changes, anything to keep in mind)</label>
              <textarea id="c-note" rows={4} value={note} onChange={(e) => setNote(e.target.value)} autoFocus
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) onAnswer('approve', note) }} />
              <div className="row">
                <button className="btn" onClick={() => onAnswer('revise', note)} disabled={!note.trim()}
                  title={note.trim() ? 'The AI redraws the concept with your changes' : 'Say what to change first'}>Request changes</button>
                <span className="spacer" />
                <button className="btn primary" onClick={() => onAnswer('approve', note)}>Approve and build</button>
              </div>
            </div>
          ) : (
            <>
              {c.note ? <p className="cnote"><b>Your answer:</b> {c.note}</p> : null}
              {onBuild && (
                <div className="canswer">
                  <label htmlFor="c-build-note" className="nm">Additional instructions (optional)</label>
                  <textarea id="c-build-note" rows={2} value={note} onChange={(e) => setNote(e.target.value)}
                    placeholder="Optional notes before building this concept…"
                    onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) onBuild(note) }} />
                  <div className="row">
                    <span className="spacer" />
                    <button className="btn primary" onClick={() => onBuild(note)}>Approve and build</button>
                  </div>
                </div>
              )}
            </>
          ))}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
