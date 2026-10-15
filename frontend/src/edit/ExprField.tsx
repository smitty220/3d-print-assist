// A number box that also takes a parameter name or a formula ("height * 0.8"). Typing a name suggests parameters; the
// value the formula gives shows underneath. A plain number commits as a number, anything else as a formula.
import { useEffect, useRef, useState } from 'react'
import { api, type ParamNode } from '../api'
import { fmtValue } from '../tree/SpecTree'

export const isNumber = (t: string) => t.trim() !== '' && Number.isFinite(Number(t))

/** The identifier being typed at the caret, and where it starts. */
export function tokenAt(text: string, caret: number): { word: string; start: number } {
  let i = caret
  while (i > 0 && /[A-Za-z0-9_]/.test(text[i - 1])) i--
  const word = text.slice(i, caret)
  return { word: /^[A-Za-z_]/.test(word) ? word : '', start: i }
}

interface Props {
  id: string
  label: string
  value: number | string | undefined // the literal value (shown when there is no formula)
  expr?: string // the formula, when one drives this value
  unit?: string
  params: ParamNode[]
  onCommit: (v: number | string) => void
  autoFocus?: boolean
}

export function ExprField({ id, label, value, expr, unit, params, onCommit, autoFocus }: Props) {
  const initial = expr ?? (typeof value === 'number' ? String(Math.round(value * 10000) / 10000) : String(value ?? ''))
  const [text, setText] = useState(initial)
  const [result, setResult] = useState<{ value?: number; error?: string } | null>(null)
  const [caret, setCaret] = useState(0)
  const [focus, setFocus] = useState(false)
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => setText(initial), [initial])

  useEffect(() => {
    if (!text.trim() || isNumber(text)) { setResult(null); return }
    let live = true // a reply for text that has since changed is dropped
    const t = setTimeout(() => api.evalExpr(text).then((r) => { if (live) setResult(r) }).catch(() => { if (live) setResult(null) }), 200)
    return () => { live = false; clearTimeout(t) }
  }, [text])

  const { word, start } = tokenAt(text, caret)
  const numeric = params.filter((p) => p.quantity !== 'text')
  const sugs = focus && word ? numeric.filter((p) => p.name.startsWith(word) && p.name !== word).slice(0, 6) : []

  const pick = (name: string) => {
    const next = text.slice(0, start) + name + text.slice(caret)
    setText(next)
    const at = start + name.length
    requestAnimationFrame(() => { ref.current?.setSelectionRange(at, at); setCaret(at) })
  }
  const commit = () => {
    if (text === initial) return
    if (isNumber(text)) onCommit(Number(text))
    else if (text.trim()) onCommit(text.trim())
  }

  return (
    <div className="xfield">
      <label htmlFor={id}>{label}{unit ? <span className="nm"> ({unit})</span> : null}</label>
      <div className="xwrap">
        <input id={id} ref={ref} value={text} autoFocus={autoFocus} spellCheck={false} autoComplete="off"
          className={expr || (!isNumber(text) && text) ? 'formula' : ''}
          onChange={(e) => { setText(e.target.value); setCaret(e.target.selectionStart ?? e.target.value.length) }}
          onKeyUp={(e) => setCaret((e.target as HTMLInputElement).selectionStart ?? 0)}
          onClick={(e) => setCaret((e.target as HTMLInputElement).selectionStart ?? 0)}
          onFocus={() => setFocus(true)} onBlur={() => { setTimeout(() => setFocus(false), 150); commit() }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { if (sugs.length) { e.preventDefault(); pick(sugs[0].name) } else commit() }
            if (e.key === 'Escape') setText(initial)
          }} />
        {sugs.length > 0 && (
          <ul className="sugs" role="listbox" aria-label="Parameters">
            {sugs.map((p) => (
              <li key={p.name} role="option" aria-selected={false}>
                <button type="button" onMouseDown={(e) => { e.preventDefault(); pick(p.name) }}>
                  <span className="pn">{p.name}</span><span className="pv">{fmtValue(p)}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      {result && <div className={`xres${result.error ? ' bad' : ''}`}>{result.error ?? `= ${Math.round((result.value ?? 0) * 1000) / 1000}${unit ? ` ${unit}` : ''}`}</div>}
    </div>
  )
}
