// The specification tree (as CATIA's): parameters and relations, sketches, bodies, and the build history in order with
// the end-of-part bar. Each node shows its update state. Click a parameter to light up what it drives; click a sketch or
// feature to light up the parameters it reads. Double-click to edit. Drag the bar onto a feature to roll back to it.
import { Fragment, useState } from 'react'
import type { ParamNode, Tree } from '../api'
import { Icon, StatusIcon } from '../ui/icons'

export type NodeKey = string // "param:h" | "sketch:S1" | "feature:F1" | "body:B1" | "ref:R1"

export function fmtValue(p: Pick<ParamNode, 'quantity' | 'value'>): string {
  if (p.value == null) return '—'
  if (typeof p.value === 'string') return `"${p.value}"`
  const v = Math.round(p.value * 1000) / 1000
  if (p.quantity === 'length') return `${v} mm`
  if (p.quantity === 'angle') return `${v}°`
  return String(v)
}

interface Props {
  tree: Tree
  selected: NodeKey | null
  onSelect: (k: NodeKey | null) => void
  onEdit: (k: NodeKey) => void
  onRollback: (index: number | null) => void
  onToggleRef: (id: string, visible: boolean) => void
}

export function SpecTree({ tree, selected, onSelect, onEdit, onRollback, onToggleRef }: Props) {
  const [open, setOpen] = useState<Record<string, boolean>>({ params: true, sketches: true, bodies: true, refs: true, history: true })
  const [dragging, setDragging] = useState(false)
  const sel = selected ?? ''
  // What lights up: a parameter's users, or the parameters a sketch / feature reads.
  const lit = new Map<string, string>()
  if (sel.startsWith('param:')) {
    const p = tree.params.find((x) => `param:${x.name}` === sel)
    p?.users.forEach((u) => lit.set(u, 'uses it'))
  } else if (sel.startsWith('feature:') || sel.startsWith('sketch:')) {
    const n = sel.startsWith('feature:') ? tree.features.find((f) => `feature:${f.id}` === sel) : tree.sketches.find((s) => `sketch:${s.id}` === sel)
    n?.inputs.forEach((p) => lit.set(`param:${p}`, 'input'))
    if (sel.startsWith('feature:')) {
      const f = tree.features.find((x) => `feature:${x.id}` === sel)
      Object.values(f?.refs ?? {}).forEach((r) => lit.set(r.startsWith('S') ? `sketch:${r}` : `feature:${r.split('.')[0]}`, 'input'))
    }
  }
  const row = (key: NodeKey, depth: number, body: React.ReactNode, extra?: { dim?: boolean; drop?: () => void }) => (
    <li key={key} role="treeitem" aria-selected={sel === key}>
      <button className={`node${sel === key ? ' sel' : ''}${lit.has(key) ? ' lit' : ''}${extra?.dim ? ' dim' : ''}`}
        style={{ paddingLeft: 8 + depth * 14 }} onClick={() => onSelect(key)} onDoubleClick={() => onEdit(key)}
        onDragOver={extra?.drop ? (e) => { e.preventDefault() } : undefined} onDrop={extra?.drop ? (e) => { e.preventDefault(); extra.drop!() } : undefined}>
        {body}
        {lit.has(key) && <span className="chip">{lit.get(key)}</span>}
      </button>
    </li>
  )
  const folder = (id: string, label: string, n: number) => (
    <li key={id}>
      <button className="node folder" onClick={() => setOpen({ ...open, [id]: !open[id] })} aria-expanded={open[id]}>
        <span className="caret">{open[id] ? '▾' : '▸'}</span><Icon name="folder" size={15} color="var(--muted)" />
        <span className="lb">{label}</span><span className="nm">{n}</span>
      </button>
    </li>
  )
  const bar = tree.rollback ?? tree.features.length
  const plain = tree.params.filter((p) => !p.relation)
  const rel = tree.params.filter((p) => p.relation)
  const paramRow = (p: ParamNode) => row(`param:${p.name}`, 1, <>
    <Icon name={p.relation ? 'relation' : 'param'} size={15} color="var(--accent)" />
    <span className="lb">{p.name}{p.relation && <span className="nm"> = {p.expr}</span>}</span>
    <span className={`val${p.status === 'stale' ? ' edited' : ''}`}>{fmtValue(p)}</span>
    {p.status !== 'ok' && <StatusIcon status={p.status} />}
  </>)

  return (
    <ul className="tree" role="tree" aria-label={tree.name}>
      <li><div className="node root"><Icon name="part" size={16} color="var(--accent)" /><span className="lb">{tree.name}</span></div></li>
      {folder('params', 'Parameters', tree.params.length)}
      {open.params && plain.map(paramRow)}
      {open.params && rel.length > 0 && <li><div className="node sub" style={{ paddingLeft: 22 }}>Relations</div></li>}
      {open.params && rel.map(paramRow)}
      {folder('sketches', 'Sketches', tree.sketches.length)}
      {open.sketches && tree.sketches.map((s) => row(`sketch:${s.id}`, 1, <>
        <Icon name="sketch" size={15} color="var(--sketch)" />
        <span className="lb">{s.id}<span className="nm"> · {s.name}{s.placement.kind === 'face' ? ' (on a face)' : ` (${s.placement.plane})`}</span></span>
        {s.dof != null && <span className="nm">{s.dof === 0 ? 'fixed' : `${s.dof} dof`}</span>}
        <StatusIcon status={s.status} />
      </>))}
      {folder('bodies', 'Bodies', tree.bodies.length)}
      {open.bodies && tree.bodies.map((b) => row(`body:${b.id}`, 1, <>
        <span className="sw" style={{ background: b.color ?? 'var(--body)' }} />
        <span className="lb">{b.name}<span className="nm"> · {b.features.length + b.fed_by.length} features{b.fed_by.length ? ` (from ${b.fed_by.join(', ')})` : ''}</span></span>
        {b.filament != null && <span className="chip slotchip" title={`Prints in filament slot ${b.filament}`}>{b.filament}</span>}
        <StatusIcon status={b.status} />
      </>))}
      {tree.references.length > 0 && folder('refs', 'References', tree.references.length)}
      {open.refs && tree.references.map((r) => row(`ref:${r.id}`, 1, <>
        <span className="sw ghost" style={{ background: r.color }} />
        <span className={`lb${r.visible ? '' : ' hidden'}`}>{r.id}<span className="nm"> · {r.name}{r.plate ? ` (plate ${r.plate})` : ''}</span></span>
        <span role="button" tabIndex={0} className={`eyebtn${r.visible ? ' on' : ''}`} aria-pressed={r.visible}
          title={r.visible ? 'Hide reference' : 'Show reference'} aria-label={r.visible ? `Hide ${r.name}` : `Show ${r.name}`}
          onClick={(e) => { e.stopPropagation(); onToggleRef(r.id, !r.visible) }}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); onToggleRef(r.id, !r.visible) } }}>
          <Icon name="eye" size={14} />
        </span>
      </>))}
      {folder('history', 'History', tree.features.length)}
      {open.history && tree.features.map((f, i) => (
        <Fragment key={f.id}>
          {i === bar && <Bar onDouble={() => onRollback(null)} setDragging={setDragging} />}
          {row(`feature:${f.id}`, 1, <>
            <Icon name={['pad', 'plate', 'profile_plate', 'sketch_pocket', 'profile_cut'].includes(f.kind) ? 'extrude' : f.kind} size={15} />
            <span className="lb">{f.id}<span className="nm"> · {f.label}{tree.bodies.length > 1 ? ` → ${tree.bodies.find((b) => b.id === f.body)?.name}` : ''}</span></span>
            {f.locked && <Icon name="lock" size={13} color="var(--accent)" label="Locked" />}
            <StatusIcon status={f.status} />
          </>, { dim: i >= bar || f.suppressed, drop: dragging ? () => onRollback(i + 1) : undefined })}
        </Fragment>
      ))}
      {open.history && bar >= tree.features.length && <Bar onDouble={() => onRollback(null)} setDragging={setDragging} />}
    </ul>
  )
}

function Bar({ onDouble, setDragging }: { onDouble: () => void; setDragging: (b: boolean) => void }) {
  return (
    <li>
      <div className="endbar" draggable onDragStart={() => setDragging(true)} onDragEnd={() => setDragging(false)} onDoubleClick={onDouble}
        title="End of part: drag onto a feature to roll back to it; double-click to bring everything back">
        <span className="rule" /><span>End of part</span>
      </div>
    </li>
  )
}
