import { useEffect, useState } from 'react'
import { fromDisplay, toDisplay, UNIT_LABEL, type Quantity, type Units } from './units'

const DIGITS: Record<Units, Record<Quantity, number>> = {
  in: { length: 3, force: 1, moment: 1, mass: 3, ratio: 2, stress: 1 },
  mm: { length: 2, force: 1, moment: 0, mass: 3, ratio: 2, stress: 0 },
}

interface NumFieldProps {
  id: string
  label: string
  /** Internal SI value (mm, N, N·mm, kg) or a plain ratio. */
  value: number | null
  quantity: Quantity
  units: Units
  min?: number
  allowEmpty?: boolean
  hideUnit?: boolean
  onCommit: (v: number | null) => void
}

/** A number input that shows a value in display units and commits the SI value on blur or Enter. */
export function NumField({ id, label, value, quantity, units, min, allowEmpty, hideUnit, onCommit }: NumFieldProps) {
  const fmt = (v: number | null) => (v == null ? '' : String(+toDisplay(v, quantity, units).toFixed(DIGITS[units][quantity])))
  const [text, setText] = useState(fmt(value))
  useEffect(() => setText(fmt(value)), [value, units]) // eslint-disable-line react-hooks/exhaustive-deps
  const unit = hideUnit ? '' : UNIT_LABEL[units][quantity]

  function commit() {
    if (text.trim() === '') {
      if (allowEmpty) onCommit(null)
      else setText(fmt(value))
      return
    }
    const n = Number(text)
    if (!Number.isFinite(n) || (min != null && n < min)) {
      setText(fmt(value))
      return
    }
    const si = fromDisplay(n, quantity, units)
    if (value == null || Math.abs(si - value) > 1e-9) onCommit(si)
  }

  return (
    <div className="numfield">
      <label htmlFor={id}>{label}{unit && <i> {unit}</i>}</label>
      <input id={id} type="number" value={text} onChange={(e) => setText(e.target.value)} onBlur={commit}
        onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />
    </div>
  )
}

interface VecFieldProps {
  id: string
  label: string
  value: [number, number, number]
  quantity: Quantity
  units: Units
  onCommit: (v: [number, number, number]) => void
}

/** Three NumFields (X, Y, Z) in a row. */
export function VecField({ id, label, value, quantity, units, onCommit }: VecFieldProps) {
  const set = (i: number, v: number | null) => {
    if (v == null) return
    const next = [...value] as [number, number, number]
    next[i] = v
    onCommit(next)
  }
  return (
    <fieldset className="vecfield">
      <legend>{label}{UNIT_LABEL[units][quantity] && <i> {UNIT_LABEL[units][quantity]}</i>}</legend>
      {(['X', 'Y', 'Z'] as const).map((axis, i) => (
        <NumField key={axis} id={`${id}-${axis}`} label={axis} value={value[i]} quantity={quantity} units={units} hideUnit onCommit={(v) => set(i, v)} />
      ))}
    </fieldset>
  )
}
