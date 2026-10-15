// The floating edit dialogs over the viewport: a feature's parameters, references and picks; a parameter's value.
// Every field commits as you leave it (in manual mode that marks the model out of date until you update).
import { useEffect, useState } from 'react'
import { api, type FeatureNode, type FeatureSpec, type FontInfo, type ParamNode, type ParamSpec, type Reference, type ReferenceInfo, type Tree } from '../api'
import { fmtValue } from '../tree/SpecTree'
import { Icon, StatusIcon } from '../ui/icons'
import { ExprField } from './ExprField'

const UNIT: Partial<Record<ParamSpec['quantity'], string>> = { length: 'mm', angle: '°' }
const PICKS: Record<string, 'faces' | 'edges'> = { shell: 'faces', text: 'faces', split: 'faces', fillet: 'edges', chamfer: 'edges' }
const PICK_HELP: Record<string, string> = {
  shell: 'Pick the faces to leave open (none: a closed hollow part).',
  text: 'Pick the face the text goes on (none: the text stands on the bed).',
  fillet: 'Pick the edges to round, or give two face groups below.',
  chamfer: 'Pick the edges to bevel, or give two face groups below.',
  split: 'Optionally pick a flat face to split along (none: the plane below).',
}
// Text parameters that name a body; "new" asks for a new body.
const BODY_PARAMS = new Set(['into', 'tool'])

interface FeatureProps {
  feature: FeatureNode
  spec: FeatureSpec
  tree: Tree
  fonts: FontInfo[]
  picking: 'faces' | 'edges' | null
  onPick: (kind: 'faces' | 'edges' | null) => void
  onApply: (patch: Record<string, unknown>) => void
  onDelete: () => void
  onEditSketch: (sid: string) => void
  onClose: () => void
}

export function FeatureDialog({ feature: f, spec, tree, fonts, picking, onPick, onApply, onDelete, onEditSketch, onClose }: FeatureProps) {
  const textParams = tree.params.filter((p) => p.quantity === 'text')
  const earlier = tree.features.slice(0, f.index)
  const pickKind = PICKS[f.kind]
  const picks = pickKind === 'edges' ? f.edges : f.faces
  const setParam = (name: string, v: unknown) => onApply({ params: { [name]: v } })

  const field = (s: ParamSpec) => {
    const id = `f-${f.id}-${s.name}`
    const v = f.params[s.name]
    if (['length', 'angle', 'ratio', 'count'].includes(s.quantity)) {
      return <ExprField key={s.name} id={id} label={s.label} value={v as number} expr={f.exprs[s.name]} unit={UNIT[s.quantity]}
        params={tree.params} onCommit={(x) => setParam(s.name, x)} />
    }
    if (s.quantity === 'choice') {
      return (
        <div key={s.name} className="xfield"><label htmlFor={id}>{s.label}</label>
          <select id={id} value={String(v)} onChange={(e) => setParam(s.name, e.target.value)}>
            {s.choices.map((c) => <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>)}
          </select>
        </div>
      )
    }
    if (s.quantity === 'bool') {
      return <label key={s.name} className="check"><input type="checkbox" checked={!!v} onChange={(e) => setParam(s.name, e.target.checked)} />{s.label}</label>
    }
    if (s.quantity === 'text' && BODY_PARAMS.has(s.name)) {
      const others = tree.bodies.filter((b) => b.id !== f.body)
      return (
        <div key={s.name} className="xfield"><label htmlFor={id}>{s.label}</label>
          <select id={id} value={String(v ?? '')} onChange={(e) => setParam(s.name, e.target.value)}>
            {f.kind === 'text' && <option value="">none (in this body)</option>}
            {!v && f.kind !== 'text' && <option value="">choose a body</option>}
            {others.map((b) => <option key={b.id} value={b.id}>{b.id} · {b.name}</option>)}
            {s.name === 'into' && <option value="new">New body</option>}
          </select>
        </div>
      )
    }
    if (s.quantity === 'text' && s.name === 'font') {
      return (
        <div key={s.name} className="xfield"><label htmlFor={id}>{s.label}</label>
          <select id={id} value={String(v)} onChange={(e) => setParam(s.name, e.target.value)}>
            {fonts.map((ft) => <option key={ft.name} value={ft.name}>{ft.name}{ft.source === 'design' ? ' (design)' : ''}</option>)}
          </select>
        </div>
      )
    }
    if (s.quantity === 'text') {
      const bound = f.exprs[s.name]
      return (
        <div key={s.name} className="xfield"><label htmlFor={id}>{s.label}</label>
          <input id={id} defaultValue={String(v ?? '')} key={`${v}-${bound}`} disabled={!!bound}
            onBlur={(e) => e.target.value !== v && setParam(s.name, e.target.value)} />
          {textParams.length > 0 && (
            <select aria-label={`${s.label} from a parameter`} value={bound ?? ''} onChange={(e) => onApply(e.target.value ? { exprs: { [s.name]: e.target.value } } : { params: { [s.name]: String(v ?? '') } })}>
              <option value="">typed here</option>
              {textParams.map((p) => <option key={p.name} value={p.name}>from {p.name}</option>)}
            </select>
          )}
        </div>
      )
    }
    const loops = Array.isArray(v) ? (v as unknown[]).length : 0
    return <div key={s.name} className="xfield"><label>{s.label}</label><span className="nm">{loops} points / outlines (edited by the AI or in code)</span></div>
  }

  return (
    <section className="dialog" aria-label={`Edit ${f.id}`}>
      <div className="dhead">
        <StatusIcon status={f.status} />
        <h2>{f.id} <span className="nm">{spec.label}</span></h2>
        <span className="spacer" />
        <button className={`pill${f.locked ? ' on' : ''}`} onClick={() => onApply({ locked: !f.locked })}
          title={f.locked ? 'Locked: the AI will not change it' : 'Lock it so the AI leaves it as it is'}>
          <Icon name="lock" size={13} />{f.locked ? 'Locked' : 'Lock'}
        </button>
        <button className="x" onClick={onClose} aria-label="Close">×</button>
      </div>
      <div className="dbody">
        {(f.error || (f.status !== 'ok' && f.message)) && <div className={`dmsg ${f.status}`}>{f.error ?? f.message}</div>}
        <p className="note">{spec.summary}</p>
        {spec.refs.map((r) => {
          const id = `r-${f.id}-${r.role}`
          if (r.target === 'sketch') {
            return (
              <div key={r.role} className="xfield"><label htmlFor={id}>{r.label}</label>
                <div className="row">
                  <select id={id} value={f.refs[r.role] ?? ''} onChange={(e) => onApply({ refs: { [r.role]: e.target.value } })}>
                    <option value="">choose a sketch</option>
                    {tree.sketches.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.name}</option>)}
                  </select>
                  {f.refs[r.role] && <button className="btn" onClick={() => onEditSketch(f.refs[r.role])}>Edit sketch</button>}
                </div>
              </div>
            )
          }
          const options = earlier.flatMap((e) => (r.kinds.length && !r.kinds.includes(e.kind) ? [] : r.kinds.length ? [e.id] : [e.id, `${e.id}.start`, `${e.id}.end`, `${e.id}.side`, `${e.id}.top`, `${e.id}.inner`]))
          return (
            <div key={r.role} className="xfield"><label htmlFor={id}>{r.label}</label>
              <input id={id} list={`${id}-l`} defaultValue={f.refs[r.role] ?? ''} key={f.refs[r.role]}
                onBlur={(e) => e.target.value !== (f.refs[r.role] ?? '') && onApply({ refs: { [r.role]: e.target.value } })} />
              <datalist id={`${id}-l`}>{options.map((o) => <option key={o} value={o} />)}</datalist>
            </div>
          )
        })}
        {pickKind && (
          <div className="picks">
            <div className="row">
              <span>{picks.length} {pickKind} picked</span>
              <span className="spacer" />
              <button className={`btn${picking === pickKind ? ' active' : ''}`} onClick={() => onPick(picking === pickKind ? null : pickKind)}>
                {picking === pickKind ? 'Done picking' : `Pick ${pickKind} on the part`}
              </button>
              {picks.length > 0 && <button className="btn" onClick={() => onApply({ [pickKind]: [] })}>Clear</button>}
            </div>
            <p className="note">{picking === pickKind ? 'Click on the part: a click adds one, clicking it again takes it off.' : PICK_HELP[f.kind]}</p>
          </div>
        )}
        <div className="fields">{spec.params.map(field)}</div>
        {Object.keys(f.driven).length > 0 && <p className="note">Uses: {Object.entries(f.driven).map(([k, v]) => `${k} ${v}`).join(', ')}</p>}
      </div>
      <div className="dfoot">
        <label className="check"><input type="checkbox" checked={f.suppressed} onChange={(e) => onApply({ suppressed: e.target.checked })} />Suppress</label>
        <span className="spacer" />
        <button className="btn danger" onClick={onDelete}>Delete</button>
        <button className="btn primary" onClick={onClose}>OK</button>
      </div>
    </section>
  )
}

interface ParamProps {
  param: ParamNode | null // null: a new parameter
  tree: Tree
  onSave: (p: { name: string; expr: string; quantity: string; comment: string }) => void
  onDelete: () => void
  onClose: () => void
}

export function ParamDialog({ param, tree, onSave, onDelete, onClose }: ParamProps) {
  const [name, setName] = useState(param?.name ?? '')
  const [quantity, setQuantity] = useState<string>(param?.quantity ?? 'length')
  const [expr, setExpr] = useState(param?.expr ?? '')
  const [comment, setComment] = useState(param?.comment ?? '')
  const others = tree.params.filter((p) => p.name !== param?.name)
  const unit = quantity === 'length' ? 'mm' : quantity === 'angle' ? '°' : undefined
  const save = (e?: string) => {
    const x = e ?? expr
    if (!name.trim() || !String(x).trim()) return
    onSave({ name: name.trim(), expr: String(x), quantity, comment })
  }
  return (
    <section className="dialog" aria-label={param ? `Parameter ${param.name}` : 'New parameter'}>
      <div className="dhead">
        {param && <StatusIcon status={param.status} />}
        <h2>{param ? `Parameter ${param.name}` : 'New parameter'}</h2>
        <span className="spacer" />
        <button className="x" onClick={onClose} aria-label="Close">×</button>
      </div>
      <div className="dbody">
        {param?.error && <div className="dmsg fail">{param.error}</div>}
        {!param && (
          <div className="xfield"><label htmlFor="p-name">Name</label>
            <input id="p-name" autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="height" spellCheck={false} />
          </div>
        )}
        <div className="xfield"><label htmlFor="p-q">Kind</label>
          <select id="p-q" value={quantity} onChange={(e) => setQuantity(e.target.value)}>
            {['length', 'angle', 'ratio', 'count', 'bool', 'text'].map((q) => <option key={q} value={q}>{q}</option>)}
          </select>
        </div>
        {quantity === 'text' ? (
          <div className="xfield"><label htmlFor="p-text">Text</label>
            <input id="p-text" value={expr} onChange={(e) => setExpr(e.target.value)} onBlur={() => param && save()} />
          </div>
        ) : (
          <ExprField id="p-expr" label="Value or formula" value={param?.value as number | undefined} expr={param?.relation ? param.expr : undefined}
            unit={unit} params={others} autoFocus={!!param}
            onCommit={(v) => { setExpr(String(v)); if (param) save(String(v)) }} />
        )}
        <div className="xfield"><label htmlFor="p-c">Note</label>
          <input id="p-c" value={comment} onChange={(e) => setComment(e.target.value)} onBlur={() => param && save()} />
        </div>
        {param && (
          <div className="note">
            {param.users.length ? <>Drives: {param.users.map((u) => u.split(':')[1]).join(', ')}</> : 'Not used yet.'}
            {param.value != null && <> · now {fmtValue(param)}</>}
          </div>
        )}
      </div>
      <div className="dfoot">
        {param && <button className="btn danger" onClick={onDelete} disabled={param.users.length > 0}
          title={param.users.length ? 'In use: change what uses it first' : undefined}>Delete</button>}
        <span className="spacer" />
        {!param && <button className="btn primary" onClick={() => save()} disabled={!name.trim() || !expr.trim()}>Add</button>}
        {param && <button className="btn primary" onClick={onClose}>OK</button>}
      </div>
    </section>
  )
}

interface RefProps {
  reference: Reference
  onSet: (patch: Partial<Reference>) => void
  onDelete: () => void
  onClose: () => void
}

/** A reference's placement, plate and colour, with its size and parts (what the AI sees with list_references). */
export function ReferenceDialog({ reference: r, onSet, onDelete, onClose }: RefProps) {
  const [info, setInfo] = useState<ReferenceInfo | null>(null)
  const placement = JSON.stringify([r.plate, r.place, r.x, r.y, r.z, r.rotate_z, r.scale])
  useEffect(() => {
    let live = true
    api.references().then((all) => { if (live) setInfo(all.find((x) => x.id === r.id) ?? null) }).catch(() => {})
    return () => { live = false }
  }, [r.id, placement])
  const num = (key: 'x' | 'y' | 'z' | 'rotate_z' | 'scale', label: string, unit?: string) => (
    <div className="xfield" key={key}><label htmlFor={`r-${key}`}>{label}{unit ? <span className="nm"> ({unit})</span> : null}</label>
      <input id={`r-${key}`} type="number" step="any" defaultValue={r[key]} key={`${r.id}-${key}-${r[key]}`}
        onBlur={(e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v) && v !== r[key]) onSet({ [key]: v }) }}
        onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} />
    </div>
  )
  const size = info?.bounds?.size.map((v) => Math.round(v * 10) / 10).join(' × ')
  return (
    <section className="dialog" aria-label={`Reference ${r.id}`}>
      <div className="dhead">
        <span className="sw ghost" style={{ background: r.color }} />
        <h2>{r.id} · {r.name}</h2>
        <span className="spacer" />
        <button className="x" onClick={onClose} aria-label="Close">×</button>
      </div>
      <div className="dbody">
        <div className="note">{r.file} · shown beside the design to look at and copy features from; never printed or exported.</div>
        {info?.error && <div className="dmsg fail">{info.error}</div>}
        <div className="xfield"><label htmlFor="r-name">Name</label>
          <input id="r-name" defaultValue={r.name} key={`${r.id}-name-${r.name}`}
            onBlur={(e) => { const v = e.target.value.trim(); if (v && v !== r.name) onSet({ name: v }) }} />
        </div>
        {info?.plates && info.plates.length > 1 && (
          <div className="xfield"><label htmlFor="r-plate">Plate</label>
            <select id="r-plate" value={r.plate ?? 1} onChange={(e) => onSet({ plate: Number(e.target.value) })}>
              {info.plates.map((n, i) => <option key={i} value={i + 1}>{i + 1}: {n}</option>)}
            </select>
          </div>
        )}
        <div className="xfield"><label htmlFor="r-place">Placement</label>
          <select id="r-place" value={r.place} onChange={(e) => onSet({ place: e.target.value as Reference['place'] })}>
            <option value="on_bed">on the bed, centred</option><option value="as_in_file">as in the file</option>
          </select>
        </div>
        {num('x', 'Move X', 'mm')}{num('y', 'Move Y', 'mm')}{num('z', 'Move Z', 'mm')}
        {num('rotate_z', 'Rotate Z', '°')}{num('scale', 'Scale')}
        <div className="xfield"><label htmlFor="r-color">Colour</label>
          <input id="r-color" type="color" value={r.color} onChange={(e) => onSet({ color: e.target.value })} />
        </div>
        {info && !info.error && (
          <div className="note">
            {size} mm · {info.triangles?.toLocaleString()} triangles · {info.part_count} part{info.part_count === 1 ? '' : 's'}
            {info.parts && info.parts.length > 1 && (
              <ul className="refparts">
                {info.parts.slice(0, 80).map((p, i) => (
                  <li key={i}><span>{p.name}</span><span>{p.size.map((v) => Math.round(v * 10) / 10).join(' × ')}</span></li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
      <div className="dfoot">
        <button className="btn danger" onClick={onDelete}>Remove</button>
        <span className="spacer" />
        <button className="btn" onClick={() => onSet({ visible: !r.visible })}>{r.visible ? 'Hide' : 'Show'}</button>
        <button className="btn primary" onClick={onClose}>OK</button>
      </div>
    </section>
  )
}
