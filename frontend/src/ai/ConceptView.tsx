// A concept the AI proposes before it builds: its drawing, what it will make, what it assumed and what it asks.
// Shown large over the 3D view while it waits; the user approves it or says what to change.
import { useRef, useState } from 'react'
import type { Concept } from '../api'

const svgUrl = (svg: string) => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
const STATUS = { pending: 'Waiting for you', approved: 'Approved', revise: 'Changes asked' } as const

interface Props {
  concept: Concept
  onAnswer: (decision: 'approve' | 'revise', note: string) => void
  onClose: () => void
}

export function ConceptView({ concept: c, onAnswer, onClose }: Props) {
  const [note, setNote] = useState('')
  const pending = c.status === 'pending'
  const draw = useRef<HTMLDivElement>(null)
  // Full screen shows the drawing alone, as big as the screen allows; Esc (or the button again) comes back.
  const fullScreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void draw.current?.requestFullscreen()
  }
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
        {/* An <img>: the drawing is shown, never run, whatever it contains. */}
        <div className="cdraw" ref={draw} onDoubleClick={fullScreen} title="Double-click for full screen"><img src={svgUrl(c.svg)} alt={`Concept drawing: ${c.title}`} /></div>
        <div className="ctext">
          {c.summary && <p className="csum">{c.summary}</p>}
          {c.assumptions.length > 0 && (<><h3>Assumed</h3><ul>{c.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul></>)}
          {c.questions.length > 0 && (<><h3>Questions</h3><ul className="qs">{c.questions.map((q, i) => <li key={i}>{q}</li>)}</ul></>)}
          {pending ? (
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
          ) : c.note ? <p className="cnote"><b>Your answer:</b> {c.note}</p> : null}
        </div>
      </div>
    </section>
  )
}
