import { useEffect, useState } from 'react'
import { isInterface, type Fastener, type IntentObject, type IntentPatch, type InterfaceObject, type Support } from './api'
import { NumField, VecField } from './fields'
import { dir, len, point, toDisplay, UNIT_LABEL, type Units } from './units'

export const KIND_LABEL: Record<IntentObject['kind'], string> = {
  planar_mount: 'Planar mount',
  bolt_pattern: 'Bolt pattern',
  bore: 'Bore',
  keep_out: 'Keep-out',
  design_space: 'Design space',
  load_case: 'Load case',
}

interface Props {
  obj: IntentObject
  all: IntentObject[]
  catalog: Fastener[]
  units: Units
  onPatch: (patch: IntentPatch) => void
  onDelete: () => void
}

export function IntentProps({ obj, all, catalog, units, onPatch, onDelete }: Props) {
  const [label, setLabel] = useState(obj.label)
  useEffect(() => setLabel(obj.label), [obj.id, obj.label])
  const u = units

  return (
    <>
      <div className="ttl">
        <b>{KIND_LABEL[obj.kind]}</b>
        <span>{obj.id} · {obj.source}</span>
      </div>
      <div className="field">
        <label htmlFor="intent-label">Label</label>
        <input id="intent-label" value={label} onChange={(e) => setLabel(e.target.value)} onBlur={() => label !== obj.label && onPatch({ label })}
          onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />
      </div>
      <div className="field">
        <label htmlFor="intent-strength">Strength</label>
        <select id="intent-strength" value={obj.strength} onChange={(e) => onPatch({ strength: e.target.value as 'hard' | 'soft' })}>
          <option value="hard">hard: validator fails the part if violated</option>
          <option value="soft">soft: aimed for and reported</option>
        </select>
      </div>

      {isInterface(obj) && <SupportField obj={obj} all={all} onPatch={onPatch} />}

      {obj.kind === 'planar_mount' && (
        <dl className="kv">
          <dt>Origin</dt><dd>{point(obj.frame.origin, u)}</dd>
          <dt>Normal</dt><dd>{dir(obj.frame.z)}</dd>
          <dt>Footprint</dt><dd>{extent(obj.footprint, u)}</dd>
          <dt>Face</dt><dd>{obj.face.part} · F{obj.face.face_id}</dd>
        </dl>
      )}

      {obj.kind === 'bolt_pattern' && <BoltPatternProps obj={obj} all={all} catalog={catalog} units={u} onPatch={onPatch} />}

      {obj.kind === 'bore' && (
        <dl className="kv">
          <dt>Diameter</dt><dd>Ø{len(obj.diameter, u)}</dd>
          <dt>Length</dt><dd>{len(obj.length, u)}</dd>
          <dt>Axis</dt><dd>{dir(obj.frame.z)}</dd>
          <dt>Center</dt><dd>{point(obj.frame.origin, u)}</dd>
          <dt>Face</dt><dd>{obj.face.part} · F{obj.face.face_id}</dd>
        </dl>
      )}

      {obj.kind === 'keep_out' && <KeepOutProps clearance={obj.clearance} part={obj.part} units={u} onPatch={onPatch} />}

      {obj.kind === 'design_space' && (
        <>
          <VecField id="ds-min" label="Min corner (mount frame)" value={obj.min} quantity="length" units={u} onCommit={(min) => onPatch({ min })} />
          <VecField id="ds-max" label="Max corner (mount frame)" value={obj.max} quantity="length" units={u} onCommit={(max) => onPatch({ max })} />
          <dl className="kv">
            <dt>Size</dt><dd>{obj.max.map((hi, i) => len(hi - obj.min[i], u)).join(' × ')}</dd>
            <dt>Frame</dt><dd>origin {point(obj.frame.origin, u)}, Z {dir(obj.frame.z)}</dd>
          </dl>
          <p className="note">Z = 0 is the mount face. The generated part must stay inside this box.</p>
        </>
      )}

      {obj.kind === 'load_case' && <LoadCaseProps obj={obj} all={all} units={u} onPatch={onPatch} />}

      <button className="btn danger" onClick={onDelete}>Delete {obj.id}</button>
    </>
  )
}

function BoltPatternProps({ obj, all, catalog, units, onPatch }: { obj: Extract<IntentObject, { kind: 'bolt_pattern' }>; all: IntentObject[]; catalog: Fastener[]; units: Units; onPatch: (p: IntentPatch) => void }) {
  const f = catalog.find((c) => c.part_number === obj.fastener)
  const diameters = [...new Set(obj.holes.map((h) => h.diameter.toFixed(4)))].map(Number)
  const depth = obj.holes[0]?.depth
  const parent = all.find((o) => o.id === obj.parent)
  const clearance = f && diameters.length === 1 ? diameters[0] - f.shank_d : null

  return (
    <>
      <div className="field">
        <label htmlFor="intent-fastener">Fastener (catalog)</label>
        <select id="intent-fastener" value={obj.fastener ?? ''} onChange={(e) => e.target.value && onPatch({ fastener: e.target.value })}>
          {!obj.fastener && <option value="">Choose…</option>}
          {catalog.map((c) => (
            <option key={c.part_number} value={c.part_number}>{c.part_number}: {c.description}</option>
          ))}
        </select>
      </div>
      <div className="field">
        <label htmlFor="intent-fit">Fit</label>
        <select id="intent-fit" value={obj.fit} onChange={(e) => onPatch({ fit: e.target.value as 'floating' | 'fixed' })}>
          <option value="floating">floating (T = H − F each part)</option>
          <option value="fixed">fixed (T₁ + T₂ = H − F)</option>
        </select>
      </div>
      <dl className="kv">
        <dt>Holes</dt><dd>{obj.holes.length}x Ø{diameters.map((d) => len(d, units)).join(', Ø')}</dd>
        {f && (<><dt>Shank</dt><dd>Ø{len(f.shank_d, units)}</dd></>)}
        {clearance != null && (<><dt>Clearance</dt><dd className={clearance < 0 ? 'bad' : ''}>{len(clearance, units)} diametral{clearance < 0 ? ' (interference)' : ''}</dd></>)}
        {f && (<><dt>Hardware</dt><dd>{f.washer} (2x), {f.nut}</dd></>)}
        {depth != null && (<><dt>Mating grip</dt><dd>{len(depth, units)}</dd></>)}
        <dt>Parent</dt><dd>{parent ? `${parent.id} ${parent.label}` : 'none'}</dd>
        <dt>Face</dt><dd>{obj.face.part} · F{obj.face.face_id}</dd>
      </dl>
      <div className="sec">Hole positions (local X, Y)</div>
      <table className="holes">
        <tbody>
          {obj.holes.map((h, i) => (
            <tr key={i}><td>{i + 1}</td><td>{len(h.center[0], units)}</td><td>{len(h.center[1], units)}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="note">Hardware is drawn with an assumed 0.250 in bracket base until a part is generated. Dimensions are approximate catalog values.</p>
    </>
  )
}

function KeepOutProps({ clearance, part, units, onPatch }: { clearance: number; part: string; units: Units; onPatch: (p: IntentPatch) => void }) {
  const toUnits = (mm: number) => (units === 'in' ? mm / 25.4 : mm)
  const [v, setV] = useState(toUnits(clearance).toFixed(units === 'in' ? 3 : 2))
  useEffect(() => setV(toUnits(clearance).toFixed(units === 'in' ? 3 : 2)), [clearance, units]) // eslint-disable-line react-hooks/exhaustive-deps
  const commit = () => {
    const n = Number(v)
    if (Number.isFinite(n) && n >= 0) onPatch({ clearance: units === 'in' ? n * 25.4 : n })
  }
  return (
    <>
      <div className="field">
        <label htmlFor="ko-clearance">Clearance ({units})</label>
        <input id="ko-clearance" type="number" min={0} step={units === 'in' ? 0.01 : 0.1} value={v} onChange={(e) => setV(e.target.value)} onBlur={commit}
          onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />
      </div>
      <dl className="kv">
        <dt>Volume</dt><dd>{part}</dd>
        <dt>Source</dt><dd>imported part</dd>
      </dl>
      <p className="note">The generated part must stay this far outside the imported volume.</p>
    </>
  )
}

const SUPPORT_TEXT: Record<Support, string> = {
  auto: 'Auto',
  fixed: 'Fixed: held in all directions',
  free: 'Free',
  normal: 'Normal: face can’t move along its normal',
  pin: 'Pin: held, free to turn about the pin',
  link: 'Link: reacts along the rod only',
}

function SupportField({ obj, all, onPatch }: { obj: InterfaceObject; all: IntentObject[]; onPatch: (p: IntentPatch) => void }) {
  const options: Support[] = { planar_mount: ['auto', 'free', 'normal'], bolt_pattern: ['auto', 'fixed', 'free'], bore: ['auto', 'pin', 'link', 'fixed', 'free'] }[obj.kind] as Support[]
  const loaded = all.some((o) => o.kind === 'load_case' && o.target === obj.id)
  // Mirrors resolved_support() in the backend.
  const auto = obj.kind === 'bolt_pattern' && obj.parent && !loaded ? 'fixed' : 'free'
  const note = {
    planar_mount: 'Normal is a linear stand-in for contact with the mating face: it also resists lift-off.',
    bolt_pattern: 'Fixed holds the clamped area under each fastener. Auto fixes a pattern on the mount face unless a load case is applied to it.',
    bore: 'Auto leaves a bore free (bores are normally where loads come in). Pin: a pivot, held by the pin but free to turn about it. Link: a rod end or link, reacting only along the line from the pin to the mating part’s centre of mass. Fixed clamps the bore in every direction.',
  }[obj.kind]
  return (
    <div className="field">
      <label htmlFor="intent-support">Support (analysis)</label>
      <select id="intent-support" value={obj.support} onChange={(e) => onPatch({ support: e.target.value as Support })}>
        {options.map((o) => <option key={o} value={o}>{o === 'auto' ? `Auto (${auto})` : SUPPORT_TEXT[o]}</option>)}
      </select>
      <p className="note">{note}</p>
    </div>
  )
}

function extent(pts: [number, number][], u: Units) {
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  return `${len(Math.max(...xs) - Math.min(...xs), u)} × ${len(Math.max(...ys) - Math.min(...ys), u)}`
}

function LoadCaseProps({ obj, all, units, onPatch }: { obj: Extract<IntentObject, { kind: 'load_case' }>; all: IntentObject[]; units: Units; onPatch: (p: IntentPatch) => void }) {
  const targets = all.filter(isInterface)
  const mag = Math.hypot(...obj.force)
  const fu = UNIT_LABEL[units].force
  return (
    <>
      <div className="field">
        <label htmlFor="lc-target">Applied at</label>
        <select id="lc-target" value={obj.target ?? ''} onChange={(e) => e.target.value && onPatch({ target: e.target.value })}>
          {!obj.target && <option value="">Choose an interface…</option>}
          {targets.map((t) => <option key={t.id} value={t.id}>{t.id} {t.label}</option>)}
        </select>
      </div>
      <div className="row2">
        <div className="field">
          <label htmlFor="lc-cat">Values are</label>
          <select id="lc-cat" value={obj.category} onChange={(e) => onPatch({ category: e.target.value as 'limit' | 'ultimate' })}>
            <option value="limit">limit</option>
            <option value="ultimate">ultimate</option>
          </select>
        </div>
        <NumField id="lc-factor" label="Ult. factor" value={obj.factor} quantity="ratio" units={units} min={0.01} onCommit={(v) => v != null && onPatch({ factor: v })} />
      </div>
      <VecField id="lc-force" label="Force (global axes)" value={obj.force} quantity="force" units={units} onCommit={(force) => onPatch({ force })} />
      <VecField id="lc-moment" label="Moment (global axes)" value={obj.moment} quantity="moment" units={units} onCommit={(moment) => onPatch({ moment })} />
      <dl className="kv">
        <dt>|F| {obj.category}</dt><dd>{toDisplay(mag, 'force', units).toFixed(1)} {fu}</dd>
        <dt>|F| {obj.category === 'limit' ? 'ultimate' : 'limit'}</dt>
        <dd>{toDisplay(obj.category === 'limit' ? mag * obj.factor : mag / obj.factor, 'force', units).toFixed(1)} {fu}</dd>
      </dl>
      <p className="note">{obj.target ? 'Applied at the target interface origin: spread over the bore surface (on the side the pin bears) or the clamped area under the fasteners. Used by topology optimisation now, FEA later.' : 'Its interface was deleted. Choose a new one.'}</p>
    </>
  )
}
