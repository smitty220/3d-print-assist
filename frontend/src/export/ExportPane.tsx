// Color & Export: the printer, the filament palette (one per printer slot), which filament each body prints in,
// the variants table (sizes) and colorways (colour sets), the print checks, and the files: one now, or every
// variant in every colorway in one zip.
import { useEffect, useRef, useState } from 'react'
import type { Checks, Colorway, Op, Printer, SlicerInfo, Tree, Variant } from '../api'
import { api } from '../api'
import { Icon } from '../ui/icons'

interface Props {
  tree: Tree
  checks: Checks | null
  printers: Printer[]
  printer: string
  onPrinter: (id: string) => void
  slicers: SlicerInfo[]
  slicer: string
  onSlicer: (id: string) => void
  onAddPrinter?: (p: { name: string; bed: [number, number, number]; nozzle_dia: number; slots: number }) => Promise<void>
  onOps: (ops: Op[], message: string) => void
  preview: string | null // the colorway shown in the view
  onPreview: (id: string | null) => void
  onError: (msg: string) => void
}

const MARK = { ok: ['ok', 'var(--ok)'], warn: ['warn', 'var(--warn)'], fail: ['fail', 'var(--bad)'] } as const
type Fmt = '3mf' | 'stl' | 'step'

/** A colour swatch that commits when the picker closes, not on every drag step. */
function ColorInput({ value, label, onCommit }: { value: string; label: string; onCommit: (c: string) => void }) {
  const ref = useRef<HTMLInputElement>(null)
  const commit = useRef(onCommit)
  useEffect(() => { commit.current = onCommit })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const on = () => { if (el.value !== value) commit.current(el.value) }
    el.addEventListener('change', on)
    return () => el.removeEventListener('change', on)
  }, [value])
  return <input ref={ref} type="color" className="swatch-in" aria-label={label} defaultValue={value} key={value} />
}

/** A text box that commits on blur or Enter. */
function Field({ value, label, onCommit, placeholder, className }: { value: string; label: string; onCommit: (v: string) => void; placeholder?: string; className?: string }) {
  return (
    <input aria-label={label} className={className} defaultValue={value} key={value} placeholder={placeholder} spellCheck={false}
      onBlur={(e) => { if (e.target.value !== value) onCommit(e.target.value) }}
      onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} />
  )
}

export function ExportPane({ tree, checks, printers, printer, onPrinter, slicers, slicer, onSlicer, onAddPrinter, onOps, preview, onPreview, onError }: Props) {
  const why = tree.stale ? 'Update the model first' : tree.failed ? 'A feature failed: fix it first' : null
  const prof = printers.find((p) => p.id === printer)
  const slots = prof ? prof.nozzles.reduce((n, z) => n + z.slots, 0) : 4
  const free = Array.from({ length: slots }, (_, i) => i + 1).find((s) => !tree.filaments.some((f) => f.slot === s))
  const [fmt, setFmt] = useState<Fmt>('3mf')
  const [busy, setBusy] = useState(false)
  const [sendMenu, setSendMenu] = useState(false)
  const [sending, setSending] = useState<string | null>(null)
  const activeSlicer = slicers.find((s) => s.id === slicer) ?? slicers.find((s) => s.installed) ?? slicers[0]
  const anySlicerInstalled = slicers.some((s) => s.installed)

  async function sendOrDownload(s: SlicerInfo) {
    if (!anySlicerInstalled) {
      const a = document.createElement('a')
      a.href = api.exportUrl('3mf')
      a.download = ''
      a.click()
      return
    }
    setSending(s.name)
    try {
      await api.sendToSlicer(s.id, '3mf')
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setSending(null)
    }
  }
  const [addingPrinter, setAddingPrinter] = useState(false)
  const [pName, setPName] = useState('')
  const [pX, setPX] = useState('256')
  const [pY, setPY] = useState('256')
  const [pZ, setPZ] = useState('256')
  const [pNozzle, setPNozzle] = useState('0.4')
  const [pSlots, setPSlots] = useState('1')
  const count = Math.max(tree.variants.length, 1) * Math.max(tree.colorways.length, 1)

  async function batch() {
    setBusy(true)
    try {
      const { blob, name } = await api.exportBatch(fmt)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = name
      a.click()
      setTimeout(() => URL.revokeObjectURL(url), 5000)
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="exportpane">
      <h2>Printer</h2>
      <div className="xfield">
        <select aria-label="Printer profile" value={printer} onChange={(e) => onPrinter(e.target.value)}>
          {printers.map((p) => <option key={p.id} value={p.id}>{p.name} · {p.bed.join(' × ')} mm · {p.nozzles.reduce((n, z) => n + z.slots, 0)} slots</option>)}
        </select>
        {onAddPrinter && !addingPrinter && (
          <button className="btn sm" type="button" onClick={() => setAddingPrinter(true)}>+ Custom printer</button>
        )}
        {onAddPrinter && addingPrinter && (
          <div className="xfield" style={{ gap: 6, padding: 8, border: '1px solid var(--line)', borderRadius: 6, background: 'var(--panel-2)' }}>
            <input aria-label="Custom printer name" placeholder="Printer name (e.g. Prusa XL)" value={pName} onChange={(e) => setPName(e.target.value)} />
            <div className="row" style={{ gap: 4 }}>
              <input aria-label="Bed X (mm)" type="number" min={10} placeholder="X" value={pX} onChange={(e) => setPX(e.target.value)} style={{ width: '100%' }} />
              <input aria-label="Bed Y (mm)" type="number" min={10} placeholder="Y" value={pY} onChange={(e) => setPY(e.target.value)} style={{ width: '100%' }} />
              <input aria-label="Bed Z (mm)" type="number" min={10} placeholder="Z" value={pZ} onChange={(e) => setPZ(e.target.value)} style={{ width: '100%' }} />
            </div>
            <div className="row" style={{ gap: 4 }}>
              <input aria-label="Nozzle diameter (mm)" type="number" step="0.05" min={0.1} placeholder="Nozzle mm" value={pNozzle} onChange={(e) => setPNozzle(e.target.value)} style={{ width: '100%' }} />
              <input aria-label="Filament slots" type="number" min={1} max={16} placeholder="Slots" value={pSlots} onChange={(e) => setPSlots(e.target.value)} style={{ width: '100%' }} />
            </div>
            <div className="row" style={{ gap: 6 }}>
              <button className="btn sm primary" type="button" onClick={async () => {
                const x = Number(pX), y = Number(pY), z = Number(pZ), dia = Number(pNozzle), s = Number(pSlots)
                if (!pName.trim() || !(x > 0 && y > 0 && z > 0 && dia > 0 && s >= 1)) {
                  onError('Enter a printer name, positive bed dimensions, nozzle size, and slots >= 1.')
                  return
                }
                await onAddPrinter({ name: pName.trim(), bed: [x, y, z], nozzle_dia: dia, slots: s })
                setAddingPrinter(false); setPName('')
              }}>Save</button>
              <button className="btn sm" type="button" onClick={() => setAddingPrinter(false)}>Cancel</button>
            </div>
          </div>
        )}
      </div>

      <h2>Filaments</h2>
      <ul className="palette">
        {tree.filaments.map((f) => (
          <li key={f.slot}>
            <span className="slot" title={`Slot ${f.slot}`}>{f.slot}</span>
            <ColorInput value={f.color} label={`Slot ${f.slot} colour`} onCommit={(c) => onOps([{ op: 'set_filament', slot: f.slot, color: c }], `Slot ${f.slot} colour`)} />
            <Field className="grow" value={f.name} label={`Slot ${f.slot} name`} placeholder="name (Matte Black)"
              onCommit={(v) => onOps([{ op: 'set_filament', slot: f.slot, name: v }], `Name slot ${f.slot}`)} />
            <Field className="mat" value={f.material} label={`Slot ${f.slot} material`}
              onCommit={(v) => onOps([{ op: 'set_filament', slot: f.slot, material: v }], `Slot ${f.slot} material`)} />
            {prof && prof.nozzles.length > 1 && (
              <select aria-label={`Slot ${f.slot} nozzle`} value={f.nozzle} onChange={(e) => onOps([{ op: 'set_filament', slot: f.slot, nozzle: Number(e.target.value) }], `Slot ${f.slot} nozzle`)}>
                {prof.nozzles.map((_, i) => <option key={i} value={i}>{i === 0 ? 'Left' : i === 1 ? 'Right' : `Nozzle ${i + 1}`}</option>)}
              </select>
            )}
            <button className="x" aria-label={`Remove slot ${f.slot}`} onClick={() => onOps([{ op: 'delete_filament', slot: f.slot }], `Remove slot ${f.slot}`)}>×</button>
          </li>
        ))}
      </ul>
      {free != null
        ? <button className="btn sm" onClick={() => onOps([{ op: 'set_filament', slot: free, color: free === 1 ? '#1a1a1a' : '#c9a227' }], `Add filament slot ${free}`)}>+ Filament (slot {free})</button>
        : <p className="note">All {slots} slots of the {prof?.name ?? 'printer'} are used.</p>}

      <h2>Bodies</h2>
      <ul className="palette">
        {tree.bodies.map((b) => (
          <li key={b.id}>
            <span className="sw" style={{ background: b.color ?? 'var(--body)' }} />
            <Field className="grow" value={b.name} label={`${b.id} name`} onCommit={(v) => onOps([{ op: 'edit_body', id: b.id, name: v }], `Rename ${b.id}`)} />
            <select aria-label={`${b.name} filament`} value={b.filament ?? ''}
              onChange={(e) => onOps([{ op: 'edit_body', id: b.id, filament: e.target.value ? Number(e.target.value) : null }], `${b.name} filament`)}>
              <option value="">no filament</option>
              {tree.filaments.map((f) => <option key={f.slot} value={f.slot}>{f.slot} · {f.name || f.color}</option>)}
            </select>
          </li>
        ))}
      </ul>
      {tree.bodies.length === 1 && <p className="note">One body prints in one colour. Split it, inlay text, or add a body to print in more.</p>}

      <h2>Variants</h2>
      {tree.variants.map((v) => <VariantCard key={v.id} v={v} tree={tree} onOps={onOps} />)}
      <button className="btn sm" onClick={() => onOps([{ op: 'set_variant', name: `Variant ${tree.variants.length + 1}` }], 'Add variant')}
        disabled={!tree.params.length} title={tree.params.length ? undefined : 'Add parameters first: a variant changes their values'}>+ Variant</button>

      <h2>Colorways</h2>
      {tree.colorways.map((c) => <ColorwayCard key={c.id} c={c} tree={tree} onOps={onOps} shown={preview === c.id} onPreview={onPreview} />)}
      <button className="btn sm" disabled={!tree.filaments.length} title={tree.filaments.length ? undefined : 'Add filaments first'}
        onClick={() => onOps([{ op: 'set_colorway', name: `Colorway ${tree.colorways.length + 1}` }], 'Add colorway')}>+ Colorway</button>

      <h2>Print checks</h2>
      {checks?.stale && <p className="note">These are for the last update.</p>}
      <ul className="checks">
        {(checks?.part ?? []).map((f, i) => (
          <li key={i}>
            <Icon name={MARK[f.status][0]} size={14} color={MARK[f.status][1]} width={2} />
            <div><b>{f.title}</b><div className="nm">{f.detail}</div></div>
          </li>
        ))}
        {!checks?.part.length && <li className="nm">Nothing built yet.</li>}
      </ul>

      <h2>Export</h2>
      {why && <p className="dmsg stale">{why}</p>}
      {!!tree.print_layout?.items && (
        <p className="note">3MF exports the {tree.print_layout.plates} print plates (see the Plates tab); STL and STEP stay assembled.</p>
      )}
      <div className="exports">
        {(['3mf', 'stl', 'step'] as const).map((f) => (
          why ? <button key={f} className="btn" disabled>{f.toUpperCase()}</button>
            : <a key={f} className={`btn${f === '3mf' ? ' primary' : ''}`} href={api.exportUrl(f)} download>{f.toUpperCase()}</a>
        ))}
      </div>
      <div className="fmenu" style={{ marginTop: 6 }}>
        <div className="row" style={{ gap: 4 }}>
          <button
            type="button"
            className="btn"
            style={{ flex: 1 }}
            disabled={!!why || !!sending || (anySlicerInstalled && !activeSlicer?.installed)}
            title={!anySlicerInstalled
              ? `Download 3MF for ${activeSlicer?.name ?? 'slicer'} (headless server)`
              : activeSlicer?.installed ? `Export 3MF and open in ${activeSlicer.name}` : `${activeSlicer?.name ?? 'Slicer'} not found in /Applications`}
            onClick={() => activeSlicer && (setSendMenu(false), sendOrDownload(activeSlicer))}
          >
            {sending ? `Opening ${sending}…` : anySlicerInstalled ? `Send to ${activeSlicer?.name ?? 'Slicer'}` : `Download 3MF (${activeSlicer?.name ?? 'Slicer'})`}
          </button>
          <button
            type="button"
            className="btn"
            aria-label="Choose slicer"
            aria-expanded={sendMenu}
            disabled={!!why || !!sending}
            onClick={() => setSendMenu((v) => !v)}
          >
            ▾
          </button>
        </div>
        {sendMenu && !why && (
          <div className="fdrop" role="menu" style={{ width: '100%', top: 'auto', bottom: 'calc(100% + 4px)' }}>
            {slicers.map((s) => (
              <button
                key={s.id}
                type="button"
                role="menuitem"
                disabled={anySlicerInstalled && !s.installed}
                title={!anySlicerInstalled ? `Download 3MF for ${s.name}` : s.installed ? `Export 3MF and open in ${s.name}` : `${s.name} not found in /Applications`}
                onClick={() => {
                  setSendMenu(false)
                  onSlicer(s.id)
                  sendOrDownload(s)
                }}
              >
                <span>{s.name}</span>
                <span className="nm" style={{ marginLeft: 'auto' }}>{!anySlicerInstalled ? '3MF' : s.installed ? 'Open' : 'not installed'}</span>
              </button>
            ))}
          </div>
        )}
      </div>
      {(tree.variants.length > 0 || tree.colorways.length > 0) && (
        <div className="row batch">
          <select aria-label="Batch format" value={fmt} onChange={(e) => setFmt(e.target.value as Fmt)}>
            {(['3mf', 'stl', 'step'] as const).map((f) => <option key={f} value={f}>{f.toUpperCase()}</option>)}
          </select>
          <button className="btn primary" disabled={busy || tree.stale} onClick={batch}
            title={tree.stale ? 'Update the model first' : 'Every variant in every colorway, in one zip'}>
            {busy ? 'Building…' : `Export all ${count} (zip)`}
          </button>
        </div>
      )}
      <p className="note">3MF files are Bambu Studio projects for the printer: one object on the plate, each body a part in its
        filament slot, the palette's colours loaded. Files are also saved in the design's <code>exports</code> folder.</p>
    </div>
  )
}

function VariantCard({ v, tree, onOps }: { v: Variant; tree: Tree; onOps: Props['onOps'] }) {
  const set = (overrides: Record<string, string>, msg: string) => onOps([{ op: 'set_variant', id: v.id, overrides }], msg)
  const unused = tree.params.filter((p) => !(p.name in v.overrides))
  return (
    <div className="card">
      <div className="row">
        <Field className="grow strong" value={v.name} label="Variant name" onCommit={(name) => onOps([{ op: 'set_variant', id: v.id, name }], `Rename ${v.name}`)} />
        <button className="x" aria-label={`Delete ${v.name}`} onClick={() => onOps([{ op: 'delete_variant', id: v.id }], `Delete ${v.name}`)}>×</button>
      </div>
      {Object.entries(v.overrides).map(([name, expr]) => {
        const p = tree.params.find((x) => x.name === name)
        return (
          <div key={name} className="row ov">
            <span className="pn">{name}</span>
            <Field className="grow mono" value={expr} label={`${v.name} ${name}`} placeholder={p?.expr}
              onCommit={(x) => set({ ...v.overrides, [name]: x }, `${v.name}: ${name} = ${x}`)} />
            <span className="nm" title="The design's value">{p?.expr}</span>
            <button className="x" aria-label={`Drop ${name}`} onClick={() => { const o = { ...v.overrides }; delete o[name]; set(o, `${v.name}: drop ${name}`) }}>×</button>
          </div>
        )
      })}
      {unused.length > 0 && (
        <select aria-label={`Vary a parameter in ${v.name}`} value="" onChange={(e) => {
          const p = tree.params.find((x) => x.name === e.target.value)
          if (p) set({ ...v.overrides, [p.name]: p.expr }, `${v.name}: vary ${p.name}`)
        }}>
          <option value="">+ vary a parameter…</option>
          {unused.map((p) => <option key={p.name} value={p.name}>{p.name} ({p.expr})</option>)}
        </select>
      )}
    </div>
  )
}

function ColorwayCard({ c, tree, onOps, shown, onPreview }: { c: Colorway; tree: Tree; onOps: Props['onOps']; shown: boolean; onPreview: (id: string | null) => void }) {
  const setColor = (slot: number, color: string | null) => {
    const colors = { ...c.colors }
    if (color) colors[slot] = color
    else delete colors[slot]
    onOps([{ op: 'set_colorway', id: c.id, colors }], `${c.name}: slot ${slot}`)
  }
  return (
    <div className={`card${shown ? ' shown' : ''}`}>
      <div className="row">
        <Field className="grow strong" value={c.name} label="Colorway name" onCommit={(name) => onOps([{ op: 'set_colorway', id: c.id, name }], `Rename ${c.name}`)} />
        <button className={`pill${shown ? ' on' : ''}`} onClick={() => onPreview(shown ? null : c.id)} title="Show these colours in the view">
          <Icon name="eye" size={13} />{shown ? 'Shown' : 'Show'}
        </button>
        <button className="x" aria-label={`Delete ${c.name}`} onClick={() => { if (shown) onPreview(null); onOps([{ op: 'delete_colorway', id: c.id }], `Delete ${c.name}`) }}>×</button>
      </div>
      <div className="row slots">
        {tree.filaments.map((f) => {
          const own = c.colors[f.slot]
          return (
            <span key={f.slot} className={`cw${own ? '' : ' same'}`} title={own ? `Slot ${f.slot}: ${own}` : `Slot ${f.slot}: as the palette`}>
              <span className="slot">{f.slot}</span>
              <ColorInput value={own ?? f.color} label={`${c.name} slot ${f.slot}`} onCommit={(x) => setColor(f.slot, x)} />
              {own && <button className="x" aria-label={`Slot ${f.slot} as the palette`} onClick={() => setColor(f.slot, null)}>×</button>}
            </span>
          )
        })}
      </div>
    </div>
  )
}
