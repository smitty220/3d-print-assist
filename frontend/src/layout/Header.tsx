import type { FileState, Printer, SlicerInfo, Tree } from '../api'
import { Icon } from '../ui/icons'
import { FileMenu, type FileAction } from './FileMenu'

export type Mode = 'part' | 'sketch' | 'export' | 'plates'

interface Props {
  tree: Tree | null
  mode: Mode
  onMode: (m: Mode) => void
  onUpdate: () => void
  onUpdateMode: (m: 'manual' | 'auto') => void
  updating: boolean
  file: FileState
  onFile: (a: FileAction) => void
  onUndo: () => void
  onRedo: () => void
  printers: Printer[]
  printer: string
  onPrinter: (id: string) => void
  slicers: SlicerInfo[]
  slicer: string
  onSlicer: (id: string) => void
}

export function Header({
  tree, mode, onMode, onUpdate, onUpdateMode, updating, file, onFile, onUndo, onRedo,
  printers, printer, onPrinter, slicers, slicer, onSlicer,
}: Props) {
  const n = tree?.pending_count ?? 0
  const manual = tree?.mode !== 'auto'
  const activePrinter = printers.find((p) => p.id === printer)
  return (
    <header className="hdr">
      <div className="brand">
        <Icon name="part" size={20} color="var(--accent)" width={2} />
        <b>3D Print Assist</b>
        <FileMenu file={file} onAction={onFile} onUndo={onUndo} onRedo={onRedo} />
        <span className="sep">/</span>
        <span className="docname" title={file.dirty ? 'Unsaved changes' : 'Saved'}>
          {file.name}{file.dirty && <span className="dot" aria-label="unsaved"> •</span>}
        </span>
      </div>
      <nav className="tabs" aria-label="Mode">
        {(['part', 'sketch', 'export', 'plates'] as Mode[]).map((m) => (
          <button key={m} aria-current={mode === m ? 'page' : undefined} className={`tab ${m}`} onClick={() => onMode(m)}>
            {m === 'part' ? 'Part' : m === 'sketch' ? 'Sketch' : m === 'export' ? 'Color & Export' : 'Plates'}
          </button>
        ))}
      </nav>
      {(printers.length > 0 || slicers.length > 0) && (
        <div className="hdr-setup" aria-label="Printer and slicer">
          {printers.length > 0 && (
            <label className="hdr-pick" title={activePrinter
              ? `${activePrinter.name} · ${activePrinter.bed.join(' × ')} mm · ${activePrinter.nozzles.reduce((sum, z) => sum + z.slots, 0)} slots`
              : 'Target printer'}>
              <span className="hdr-lbl">Printer</span>
              <select aria-label="Printer" value={printer} onChange={(e) => onPrinter(e.target.value)}>
                {printers.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
            </label>
          )}
          {slicers.length > 0 && (
            <label className="hdr-pick" title="Target slicer">
              <span className="hdr-lbl">Slicer</span>
              <select aria-label="Slicer" value={slicer} onChange={(e) => onSlicer(e.target.value)}>
                {slicers.map((s) => (
                  <option key={s.id} value={s.id}>{s.name}{s.installed ? '' : ' (not installed)'}</option>
                ))}
              </select>
            </label>
          )}
        </div>
      )}
      <div className="spacer" />
      <div className="seg" role="group" aria-label="Update mode">
        <button aria-pressed={manual} onClick={() => onUpdateMode('manual')} title="Edits wait for Update">Manual</button>
        <button aria-pressed={!manual} onClick={() => onUpdateMode('auto')} title="Every edit rebuilds">Auto</button>
      </div>
      <button className={`update${n ? ' due' : tree?.failed ? ' failed' : ''}`} onClick={onUpdate} disabled={updating || (!n && !tree?.failed)}
        title={tree?.failed && !n ? 'The last update stopped at a failed feature; fix it, or update again' : 'Update geometry (Ctrl+B)'}>
        <Icon name={n ? 'bolt' : tree?.failed ? 'fail' : 'ok'} size={16} fill={n ? 'currentColor' : 'none'} width={1.9} />
        {updating ? 'Updating…' : n ? `Update (${n})` : tree?.failed ? 'Update failed' : 'Up to date'}
      </button>
    </header>
  )
}
