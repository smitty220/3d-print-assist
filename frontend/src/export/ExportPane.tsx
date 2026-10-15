// Color & Export: the printer, the print checks, and the files. Colours, filament slots and variants come with M2.
import type { Checks, Printer, Tree } from '../api'
import { api } from '../api'
import { Icon } from '../ui/icons'

interface Props {
  tree: Tree
  checks: Checks | null
  printers: Printer[]
  printer: string
  onPrinter: (id: string) => void
}

const MARK = { ok: ['ok', 'var(--ok)'], warn: ['warn', 'var(--warn)'], fail: ['fail', 'var(--bad)'] } as const

export function ExportPane({ tree, checks, printers, printer, onPrinter }: Props) {
  const why = tree.stale ? 'Update the model first' : tree.failed ? 'A feature failed: fix it first' : null
  return (
    <div className="exportpane">
      <h2>Printer</h2>
      <div className="xfield">
        <select aria-label="Printer profile" value={printer} onChange={(e) => onPrinter(e.target.value)}>
          {printers.map((p) => <option key={p.id} value={p.id}>{p.name} · {p.bed.join(' × ')} mm · {p.nozzles.reduce((n, z) => n + z.slots, 0)} slots</option>)}
        </select>
      </div>
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
      <div className="exports">
        {(['3mf', 'stl', 'step'] as const).map((f) => (
          why ? <button key={f} className="btn" disabled>{f.toUpperCase()}</button>
            : <a key={f} className={`btn${f === '3mf' ? ' primary' : ''}`} href={api.exportUrl(f)} download>{f.toUpperCase()}</a>
        ))}
      </div>
      <p className="note">3MF opens in Bambu Studio placed on the plate; each body is its own object. Colours, filament slots and
        size / colourway variants arrive with milestone M2 (built from a Bambu Studio reference project). Files are also saved in
        the design's <code>exports</code> folder.</p>
    </div>
  )
}
