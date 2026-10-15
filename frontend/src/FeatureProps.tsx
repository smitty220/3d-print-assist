import { useEffect, useState } from 'react'
import type { FeaturePatch, FeatureSpec, FeatureState, IntentObject, ParamSpec, ParamValue, RuleCheck } from './api'
import { NumField } from './fields'
import { fromDisplay, len, toDisplay, type Units } from './units'

const LOCK = (
  <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><rect x="3" y="7" width="10" height="7.5" rx="1.5" fill="currentColor" /><path d="M5.2 7V5a2.8 2.8 0 0 1 5.6 0v2" fill="none" stroke="currentColor" strokeWidth="1.6" /></svg>
)
const UNLOCK = (
  <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><rect x="3" y="7" width="10" height="7.5" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="M5.2 7V5a2.8 2.8 0 0 1 5.4-1" fill="none" stroke="currentColor" strokeWidth="1.6" /></svg>
)

export function LockButton({ locked, label, onToggle }: { locked: boolean; label: string; onToggle: () => void }) {
  return (
    <button className="lockb" aria-pressed={locked} aria-label={`${locked ? 'Unlock' : 'Lock'} ${label}`}
      title={locked ? 'Locked: the AI will not change this feature' : 'Unlocked: the AI may change this feature'}
      onClick={(e) => { e.stopPropagation(); onToggle() }}>
      {locked ? LOCK : UNLOCK}
    </button>
  )
}

const ICON = { ok: '✔', warn: '⚠', fail: '✖' } as const
const CLASS = { ok: 'ok', warn: 'wn', fail: 'no' } as const

export function CheckList({ checks, units }: { checks: RuleCheck[]; units: Units }) {
  if (!checks.length) return <p className="note">No rule checks for this feature.</p>
  return (
    <div className="drv">
      {checks.map((c) => (
        <div key={c.name} className="chk">
          <span className="k">{c.name}</span>
          <span className="v">{c.value == null ? '' : isRatio(c.name) ? c.value.toFixed(2) : len(c.value, units)}</span>
          <span className={CLASS[c.status]}>{ICON[c.status]}</span>
          <span className="rule">{c.rule}</span>
        </div>
      ))}
    </div>
  )
}

function isRatio(name: string) {
  return /\/|e\/D|t\/D/.test(name)
}

function drivenText(key: string, v: number | string, units: Units) {
  if (typeof v === 'string') return v
  if (key === 'count') return String(v)
  return len(v, units)
}

const DRIVEN_LABEL: Record<string, string> = {
  pattern: 'Bolt pattern', fastener: 'Fastener', count: 'Holes', hole_d: 'Hole Ø', pin: 'Pin bore', pin_d: 'Pin Ø', gap: 'Clevis gap',
  side_clearance: 'Side clearance (each)',
}

interface Props {
  feature: FeatureState
  spec: FeatureSpec
  checks: RuleCheck[]
  objects: IntentObject[]
  features: FeatureState[]
  faceGroups: string[]
  units: Units
  onPatch: (patch: FeaturePatch) => void
  onDelete: () => void
  onEditSketch?: (id: string) => void
  /** Fillet / chamfer: picking its edges on the part. */
  pickingEdges?: boolean
  onPickEdges?: (on: boolean) => void
}

const EDGE_KINDS = ['fillet', 'chamfer']

export function FeatureProps({ feature, spec, checks, objects, features, faceGroups, units, onPatch, onDelete, onEditSketch, pickingEdges, onPickEdges }: Props) {
  const edgeOp = EDGE_KINDS.includes(feature.kind)
  const picks = feature.edges ?? []
  return (
    <>
      <div className="ttl">
        <b>{spec.label}</b>
        <span>{feature.id} · {spec.group}</span>
        <LockButton locked={feature.locked} label={feature.id} onToggle={() => onPatch({ locked: !feature.locked })} />
      </div>
      {feature.error && <p className="errbox">{feature.error}</p>}

      {Object.keys(feature.driven).length > 0 && (
        <>
          <div className="sec">Driven by intent</div>
          <dl className="kv">
            {Object.entries(feature.driven).map(([k, v]) => (
              <div key={k} className="kvrow"><dt>{DRIVEN_LABEL[k] ?? k}</dt><dd>{drivenText(k, v, units)}</dd></div>
            ))}
          </dl>
        </>
      )}

      {edgeOp && (
        <>
          <div className="sec">Edges</div>
          {picks.length ? (
            <ol className="ctl-list edge-picks">
              {picks.map((e, i) => (
                <li key={i}>
                  <span className="k">Edge {i + 1}</span>
                  <span className="v">{e.faces.filter(Boolean).join(' / ') || 'picked'} · {len(e.length, units)} {units}</span>
                  <button className="x" aria-label={`Remove edge ${i + 1}`} title="Remove this edge" onClick={() => onPatch({ edges: picks.filter((_, k) => k !== i) })}>×</button>
                </li>
              ))}
            </ol>
          ) : <p className="note">No edges picked{feature.refs.faces_a && feature.refs.faces_b ? `: blending where ${feature.refs.faces_a} meets ${feature.refs.faces_b}` : ''}.</p>}
          {onPickEdges && (
            <button className={`btn${pickingEdges ? ' primary' : ''}`} aria-pressed={!!pickingEdges} onClick={() => onPickEdges(!pickingEdges)}>
              {pickingEdges ? '✓ Done picking' : '⌖ Pick edges on the part'}
            </button>
          )}
          <p className="note">Click an edge where two faces meet. The blend carries on along edges that continue it tangentially. Edges are kept by position, so they follow upstream dimension changes.</p>
        </>
      )}

      {spec.refs.length > 0 && !(edgeOp && picks.length) && <div className="sec">{edgeOp ? 'Or by face groups' : 'References'}</div>}
      {spec.refs.map((r) => {
        if (edgeOp && picks.length) return null
        const value = feature.refs[r.role] ?? ''
        const id = `ref-${feature.id}-${r.role}`
        if (r.target === 'sketch') {
          return (
            <div className="field" key={r.role}>
              <label>{r.label}</label>
              <div className="sp-row">
                <span className="note">{value || 'none'}</span>
                {value && onEditSketch && <button className="btn" onClick={() => onEditSketch(value)}>✎ Edit sketch</button>}
              </div>
            </div>
          )
        }
        if (r.target === 'feature' && r.kinds.length === 0) {
          return (
            <div className="field" key={r.role}>
              <label htmlFor={id}>{r.label}</label>
              <select id={id} value={value} onChange={(e) => onPatch({ refs: { [r.role]: e.target.value } })}>
                {!faceGroups.includes(value) && <option value={value}>{value || 'Choose…'}</option>}
                {faceGroups.map((g) => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>
          )
        }
        const options = r.target === 'intent'
          ? objects.filter((o) => r.kinds.includes(o.kind)).map((o) => ({ id: o.id, label: `${o.id} ${o.label}` }))
          : features.filter((f) => r.kinds.includes(f.kind) && f.id !== feature.id).map((f) => ({ id: f.id, label: `${f.id} ${f.kind}` }))
        return (
          <div className="field" key={r.role}>
            <label htmlFor={id}>{r.label}</label>
            <select id={id} value={value} onChange={(e) => onPatch({ refs: { [r.role]: e.target.value } })}>
              {!options.some((o) => o.id === value) && <option value={value}>{value || 'Choose…'}</option>}
              {options.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
            </select>
          </div>
        )
      })}

      {spec.params.length > 0 && <div className="sec">Parameters</div>}
      {spec.params.map((ps) => (
        <ParamInput key={ps.name} id={`p-${feature.id}-${ps.name}`} spec={ps} value={feature.params[ps.name]} units={units}
          onCommit={(v) => onPatch({ params: { [ps.name]: v } })} />
      ))}

      <div className="sec">Rule checks</div>
      <CheckList checks={checks} units={units} />

      <p className="note">
        {feature.locked ? 'Locked: the AI leaves this feature alone in later iterations. You can still edit it.' : 'Edits rebuild the part with no AI call. Lock it to stop the AI changing it.'}
      </p>
      <div className="sp-row">
        <button className="btn" onClick={() => onPatch({ suppressed: !feature.suppressed })} aria-pressed={feature.suppressed}>
          {feature.suppressed ? 'Unsuppress' : 'Suppress'}
        </button>
        <button className="btn danger" onClick={onDelete}>Delete {feature.id}</button>
      </div>
    </>
  )
}

function ParamInput({ id, spec, value, units, onCommit }: { id: string; spec: ParamSpec; value: ParamValue; units: Units; onCommit: (v: number | string) => void }) {
  if (spec.quantity === 'profile' || spec.quantity === 'profiles') {
    const loops = (spec.quantity === 'profile' ? (value as number[][]).length ? [value as number[][]] : [] : (value as number[][][]))
    return (
      <div className="field">
        <label>{spec.label}</label>
        <p className="note">{loops.length ? loops.map((l) => `${l.length}-point outline, ${extentText(l, units)}`).join('; ') : 'none'}</p>
      </div>
    )
  }
  if (spec.quantity === 'choice') {
    return (
      <div className="field">
        <label htmlFor={id}>{spec.label}</label>
        <select id={id} value={String(value)} onChange={(e) => onCommit(e.target.value)}>
          {spec.choices.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
    )
  }
  const q = spec.quantity === 'length' ? 'length' : 'ratio'
  return <ParamSlider id={id} spec={spec} value={Number(value)} quantity={q} units={units} onCommit={onCommit} />
}

function extentText(pts: number[][], units: Units) {
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  return `${len(Math.max(...xs) - Math.min(...xs), units)} × ${len(Math.max(...ys) - Math.min(...ys), units)}`
}

/** Range slider plus exact field. The slider previews locally and commits on release, so a drag costs one rebuild. */
function ParamSlider({ id, spec, value, quantity, units, onCommit }: { id: string; spec: ParamSpec; value: number; quantity: 'length' | 'ratio'; units: Units; onCommit: (v: number) => void }) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  // The declared bounds are wide (e.g. 0.5–400 mm), so the slider covers a useful window around the value.
  const lo = toDisplay(spec.min ?? -1e6, quantity, units)
  const hi = toDisplay(spec.max ?? 1e6, quantity, units)
  const v = toDisplay(value, quantity, units)
  const w = Math.max(Math.abs(v) * 1.5, quantity === 'ratio' ? 1 : units === 'in' ? 1 : 25)
  const sMin = Math.max(lo, v - w)
  const sMax = Math.min(hi, v + w)
  const step = quantity === 'ratio' ? 0.01 : units === 'in' ? 0.005 : 0.1
  const commit = () => { if (Math.abs(draft - value) > 1e-9) onCommit(draft) }
  return (
    <div className="prm">
      <NumField id={id} label={spec.label} value={value} quantity={quantity} units={units} min={lo} onCommit={(v) => v != null && onCommit(v)} />
      <input type="range" aria-label={`${spec.label} slider`} min={sMin} max={sMax} step={step} value={toDisplay(draft, quantity, units)}
        onChange={(e) => setDraft(fromDisplay(+e.target.value, quantity, units))} onPointerUp={commit} onKeyUp={commit} onBlur={commit} />
    </div>
  )
}
