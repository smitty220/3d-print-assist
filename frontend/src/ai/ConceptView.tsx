// A concept the AI proposes before it builds: its drawing, what it will make, what it assumed and what it asks.
// Shown large over the 3D view while it waits; the user approves it or says what to change.
import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, type Concept } from '../api'

const svgUrl = (svg: string) => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
const STATUS = { pending: 'Waiting for you', approved: 'Approved', revise: 'Changes asked' } as const

interface Props {
  concept: Concept
  concepts?: Concept[]
  onSelectConcept?: (id: string) => void
  hideAnswerInput?: boolean
  onAnswer: (decision: 'approve' | 'revise', note: string) => void
  onBuild?: (note: string) => void
  onClose: () => void
}

export function ConceptView({ concept: c, concepts, onSelectConcept, hideAnswerInput, onAnswer, onBuild, onClose }: Props) {
  const [note, setNote] = useState('')
  const [expanded, setExpanded] = useState(false)
  const [split, setSplit] = useState(0.62) // fraction of cbody height for Drawing (0 = spec full, 1 = drawing full)
  const bodyRef = useRef<HTMLDivElement>(null)
  const pending = c.status === 'pending'
  const isCharacter = c.mode === 'character' || Boolean(c.image_file)
  const previewSrc = c.image_file ? api.conceptPictureUrl(c.id) : svgUrl(c.svg)

  const startDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('button')) return
    e.preventDefault()
    const body = bodyRef.current
    if (!body) return
    const rect = body.getBoundingClientRect()
    const onMove = (ev: PointerEvent) => {
      if (rect.height <= 0) return
      const next = Math.max(0, Math.min(1, (ev.clientY - rect.top) / rect.height))
      setSplit(next < 0.08 ? 0 : next > 0.92 ? 1 : next)
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

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
        {concepts && concepts.length > 1 && onSelectConcept ? (
          <select
            aria-label="Select concept"
            value={c.id}
            onChange={(e) => onSelectConcept(e.target.value)}
            style={{ fontSize: 13, fontWeight: 600, padding: '2px 6px', maxWidth: 420 }}
          >
            {concepts.map((item) => (
              <option key={item.id} value={item.id}>{item.id} · {item.title} ({STATUS[item.status]})</option>
            ))}
          </select>
        ) : (
          <h2>{c.id} · {c.title}</h2>
        )}
        <span className={`chip cstat ${c.status}`}>{STATUS[c.status]}</span>
        {(c.views && c.views.length > 0) ? (
          <span className="chip">{`${c.views.length} Views · ${c.views.join(' / ')}`}</span>
        ) : (
          isCharacter && <span className="chip">Character Design</span>
        )}
        <span className="spacer" />
        <button className="btn" onClick={fullScreen} title="Show the drawing full screen (Esc to come back; or double-click the drawing)">⛶ Full screen</button>
        <button className="x" onClick={onClose} aria-label={pending ? 'Hide for now' : 'Close'} title={pending ? 'Hide for now (it stays in the AI panel)' : 'Close'}>×</button>
      </div>
      <div className="cbody" ref={bodyRef}>
        <div
          className={`cpane cdraw-pane${split >= 0.99 ? ' is-full' : split <= 0.01 ? ' is-collapsed' : ''}`}
          style={split > 0.01 && split < 0.99 ? { flex: `${split} 1 0` } : undefined}
        >
          <div className="cpane-bar">
            <span className="cpane-title">Drawing</span>
            <span className="spacer" />
            <button
              type="button"
              className="btn sm cpane-caret"
              title={split >= 0.99 ? 'Restore split' : 'Expand drawing'}
              aria-label={split >= 0.99 ? 'Restore split' : 'Expand drawing'}
              onClick={() => setSplit(split >= 0.99 ? 0.62 : 1)}
            >
              {split >= 0.99 ? '▲' : '▼'}
            </button>
          </div>
          {split > 0.01 && (
            <div className="cdraw" onDoubleClick={fullScreen} title="Double-click for full screen">
              <img src={previewSrc} alt={`Concept drawing: ${c.title}`} />
            </div>
          )}
        </div>
        {expanded && createPortal(
          <div className="cdraw-fullscreen" role="dialog" aria-label={`Full screen drawing: ${c.title}`} onDoubleClick={() => setExpanded(false)}>
            <button type="button" className="btn cdraw-exit" onClick={() => setExpanded(false)} title="Exit full screen (Esc)">
              ✕ Exit full screen
            </button>
            <img src={previewSrc} alt={`Concept drawing: ${c.title}`} />
          </div>,
          document.body,
        )}
        <div
          className={`cpane ctext-pane${split <= 0.01 ? ' is-full' : split >= 0.99 ? ' is-collapsed' : ''}`}
          style={split > 0.01 && split < 0.99 ? { flex: `${1 - split} 1 0` } : undefined}
        >
          <div
            className="cpane-bar cpane-drag"
            onPointerDown={startDrag}
            onDoubleClick={() => setSplit(0.62)}
            title="Drag up/down to resize (double-click to reset)"
          >
            <span className="cpane-title">Spec, Assumptions &amp; Questions</span>
            <span className="spacer" />
            <button
              type="button"
              className="btn sm cpane-caret"
              title={split <= 0.01 ? 'Restore split' : 'Expand spec'}
              aria-label={split <= 0.01 ? 'Restore split' : 'Expand spec'}
              onClick={() => setSplit(split <= 0.01 ? 0.62 : 0)}
            >
              {split <= 0.01 ? '▼' : '▲'}
            </button>
          </div>
          {split < 0.99 && (
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
