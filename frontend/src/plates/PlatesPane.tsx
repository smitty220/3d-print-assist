// Plates: the print layout, worked out when this tab opens (and again after the model or the layout changes): the
// plates, the objects on each, anything that keeps them from printing, and the export of the plates as one Bambu
// Studio project. The arrangement itself is made and changed by asking Claude; nothing here edits it.
import { useEffect, useState } from 'react'
import { api, type LayoutReport, type SlicerInfo, type Tree } from '../api'
import { Icon } from '../ui/icons'

interface Props {
  tree: Tree
  plate: number
  onPlate: (n: number) => void
  slicers: SlicerInfo[]
  slicer: string
  onError: (msg: string) => void
}

export function PlatesPane({ tree, plate, onPlate, slicers, slicer, onError }: Props) {
  const [report, setReport] = useState<LayoutReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [sending, setSending] = useState(false)
  const activeSlicer = slicers.find((s) => s.id === slicer) ?? slicers.find((s) => s.installed) ?? slicers[0]
  const anySlicerInstalled = slicers.some((s) => s.installed)
  const items = tree.print_layout?.items ?? 0
  const why = tree.stale ? 'Update the model first' : tree.failed ? 'A feature failed: fix it first' : null

  useEffect(() => {
    if (!items) return
    let live = true
    api.layout().then((r) => { if (live) setReport(r) }).catch((e) => onError(e instanceof Error ? e.message : String(e)))
    return () => { live = false }
  }, [tree, items, onError])

  async function exportPlates() {
    setBusy(true)
    try {
      const { blob, name } = await api.exportFile('3mf')
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

  if (!items) {
    return (
      <div className="plates-pane">
        <h2>Print plates</h2>
        <p className="note">No plate layout yet. Ask Claude to lay the parts out on plates for your printer.</p>
      </div>
    )
  }
  const shown = report?.plates.find((p) => p.plate === plate)
  const problems = report?.problems ?? []
  return (
    <div className="plates-pane">
      <h2>Print plates</h2>
      {!report && <p className="note">Laying out…</p>}
      <ul className="plate-list" role="listbox" aria-label="Plates">
        {(report?.plates ?? []).map((p) => (
          <li key={p.plate}>
            <button role="option" aria-selected={p.plate === plate} className={`plate-btn${p.plate === plate ? ' on' : ''}`} onClick={() => onPlate(p.plate)}>
              <span className="pn">{p.plate}</span>
              <span className="pl">{p.name.replace(/^\d+\s*/, '') || `Plate ${p.plate}`}</span>
              <span className="nm">{p.objects.length}</span>
            </button>
          </li>
        ))}
      </ul>
      {shown && (
        <>
          <h2>On plate {shown.plate}</h2>
          <ul className="plate-objs">
            {shown.objects.map((o) => (
              <li key={`${o.id}.${o.copy ?? 1}`}>
                <b>{o.name}{o.copy ? ` #${o.copy}` : ''}</b>
                <span className="nm">{(o.box[3] - o.box[0]).toFixed(0)} × {(o.box[4] - o.box[1]).toFixed(0)} × {(o.box[5] - o.box[2]).toFixed(1)} mm</span>
              </li>
            ))}
            {!shown.objects.length && <li className="nm">Nothing on this plate.</li>}
          </ul>
          {shown.tower && <p className="nm">Purge tower at {shown.tower[0]}, {shown.tower[1]} (dashed).</p>}
        </>
      )}
      {problems.length > 0 && (
        <>
          <h2>Problems</h2>
          <ul className="checks">
            {problems.map((t, i) => (
              <li key={i}><Icon name="fail" size={14} color="var(--bad)" width={2} /><div>{t}</div></li>
            ))}
          </ul>
        </>
      )}
      <h2>Export</h2>
      {why && <p className="dmsg stale">{why}</p>}
      <div className="exports">
        <button className="btn primary" disabled={!!why || busy || problems.length > 0} onClick={exportPlates}>
          {busy ? 'Exporting…' : `Export ${report?.plates.length ?? tree.print_layout?.plates} plates (3MF)`}
        </button>
        {activeSlicer && anySlicerInstalled && (
          <button
            className="btn"
            disabled={!!why || sending || problems.length > 0 || !activeSlicer.installed}
            title={activeSlicer.installed ? `Export 3MF and open in ${activeSlicer.name}` : `${activeSlicer.name} not found in /Applications`}
            onClick={async () => {
              setSending(true)
              try {
                await api.sendToSlicer(activeSlicer.id, '3mf')
              } catch (e) {
                onError(e instanceof Error ? e.message : String(e))
              } finally {
                setSending(false)
              }
            }}
          >
            {sending ? `Opening ${activeSlicer.name}…` : `Send to ${activeSlicer.name}`}
          </button>
        )}
      </div>
      <p className="nm">One 3MF project: each plate ready to slice.</p>
    </div>
  )
}
