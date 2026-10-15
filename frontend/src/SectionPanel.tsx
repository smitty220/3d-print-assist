import { useState } from 'react'
import type { FaceMesh } from './api'
import { dir, len, point, type Units } from './units'
import { BASE_NORMALS, resolveSection, type SectionBase, type SectionSpec } from './viewer/sectionPlane'

const MM_PER_IN = 25.4

interface Props {
  spec: SectionSpec
  units: Units
  range: number // mm, half-range for the offset slider
  showPlane: boolean
  pickedFace: FaceMesh | undefined
  pickedLabel: string | undefined
  onChange: (spec: SectionSpec) => void
  onShowPlane: (show: boolean) => void
  onBase: (base: Exclude<SectionBase, 'face'>) => void
  onAlignToFace: () => void
  onLookAt: () => void
}

export function SectionPanel({ spec, units, range, showPlane, pickedFace, pickedLabel, onChange, onShowPlane, onBase, onAlignToFace, onLookAt }: Props) {
  const toUnits = (mm: number) => (units === 'in' ? mm / MM_PER_IN : mm)
  const fromUnits = (v: number) => (units === 'in' ? v * MM_PER_IN : v)
  const step = units === 'in' ? 0.01 : 0.1
  const { normal, point: p } = resolveSection(spec)
  const set = (patch: Partial<SectionSpec>) => onChange({ ...spec, ...patch })
  const canAlign = pickedFace && (pickedFace.type === 'plane' || pickedFace.type === 'cylinder')
  const alignLabel = pickedFace?.type === 'cylinder' ? 'Through picked hole axis' : 'At picked face'
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="secpanel" role="group" aria-label="Section plane">
      <div className="sp-head">
        <b>Section</b>
        <span>{spec.base === 'face' ? spec.source : `${spec.base} plane`}</span>
        <button className="sp-min" onClick={() => setCollapsed(!collapsed)} aria-expanded={!collapsed} aria-label={collapsed ? 'Expand section panel' : 'Collapse section panel'}>
          {collapsed ? '+' : '−'}
        </button>
      </div>
      {!collapsed && (<>

      <div className="sp-row seg" role="group" aria-label="Base plane">
        {(Object.keys(BASE_NORMALS) as Exclude<SectionBase, 'face'>[]).map((b) => (
          <button key={b} aria-pressed={spec.base === b} onClick={() => onBase(b)}>
            {b}
          </button>
        ))}
      </div>
      <button className="btn wide" disabled={!canAlign} onClick={onAlignToFace} title={canAlign ? `${alignLabel}: ${pickedLabel}` : 'Pick a planar face or a hole first'}>
        {canAlign ? alignLabel : 'Align to picked face or hole'}
      </button>

      <Slider id="sec-offset" label={`Offset (${units})`} min={toUnits(-range)} max={toUnits(range)} step={step}
        value={toUnits(spec.offset)} digits={units === 'in' ? 3 : 2} onValue={(v) => set({ offset: fromUnits(v) })} />
      <Slider id="sec-tilt-a" label="Tilt A (°)" min={-90} max={90} step={1} value={spec.tiltA} digits={1} onValue={(v) => set({ tiltA: v })} />
      <Slider id="sec-tilt-b" label="Tilt B (°)" min={-90} max={90} step={1} value={spec.tiltB} digits={1} onValue={(v) => set({ tiltB: v })} />

      <div className="sp-row">
        <button className="btn" onClick={() => set({ flip: !spec.flip })} aria-pressed={spec.flip}>Flip side</button>
        <button className="btn" onClick={() => set({ tiltA: 0, tiltB: 0, offset: 0 })}>Reset</button>
        <button className="btn" onClick={onLookAt}>Look at</button>
      </div>
      <label className="sp-check" htmlFor="sec-show">
        <input id="sec-show" type="checkbox" checked={showPlane} onChange={(e) => onShowPlane(e.target.checked)} /> Show plane
      </label>

      <dl className="kv">
        <dt>Point</dt>
        <dd>{point([p.x, p.y, p.z], units)}</dd>
        <dt>Normal</dt>
        <dd>{dir([round(normal.x), round(normal.y), round(normal.z)])}</dd>
      </dl>
      <p className="note">Drag the orange arrow to move the plane. The arrow points the direction of sight, toward the material kept. Offset {len(spec.offset, units)} {units} from the reference point.</p>
      </>)}
    </div>
  )
}

function round(v: number) {
  return Math.round(v * 1e6) / 1e6
}

interface SliderProps {
  id: string
  label: string
  min: number
  max: number
  step: number
  value: number
  digits: number
  onValue: (v: number) => void
}

function Slider({ id, label, min, max, step, value, digits, onValue }: SliderProps) {
  return (
    <div className="prm">
      <label htmlFor={`${id}-n`}>{label}</label>
      <input type="range" id={`${id}-r`} aria-label={label} min={min} max={max} step={step} value={value} onChange={(e) => onValue(+e.target.value)} />
      <input type="number" id={`${id}-n`} step={step} value={+value.toFixed(digits)} onChange={(e) => e.target.value !== '' && onValue(+e.target.value)} />
    </div>
  )
}
