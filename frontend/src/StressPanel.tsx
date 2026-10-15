// Stress analysis of the designed part in three steps: set up and check the mesh (by hand, by the best-practice rules or by
// Claude), solve and read the results (contours, margins by region), and size the part for a target margin.
import { useEffect, useState } from 'react'
import type { FemBearing, FemField, FemJob, FemMesh, FemPeakSpot, FemPeaks, MeshSetup, MeshSetupState } from './api'
import { bandColor, turbo } from './viewer/Viewer'
import { fromDisplay, len, toDisplay, type Units } from './units'

const KSI = 6.894757
export type FemQuantity = 'vm' | 'p1' | 'u'
export type FemTab = 'mesh' | 'results' | 'size'
export type PickKind = 'edge' | 'face' | 'sphere'
export type ControlRef = { kind: 'edge' | 'face' | 'sphere'; index: number }
const QTY: Record<FemQuantity, string> = { vm: 'von Mises stress', p1: 'max principal stress', u: 'displacement' }
export const BANDS = 12

/** The text on the viewport flag at a contour's peak: what, how much, which load case. */
export function peakLabel(f: FemField, caseId: string | null, units: Units): string {
  const v = f.max
  const val = f.quantity === 'u'
    ? (units === 'in' ? `${(v / 25.4).toFixed(4)} in` : `${v.toFixed(3)} mm`)
    : (units === 'in' ? `${(v / KSI).toFixed(1)} ksi` : `${v.toFixed(0)} MPa`)
  const what = f.quantity === 'vm' ? 'Peak von Mises' : f.quantity === 'p1' ? 'Peak max principal' : 'Peak displacement'
  return `${what} ${val}${caseId ? ` · ${caseId}` : ''}`
}
// Best-practice targets the mesh check marks (kept in step with the default practices in backend fem/controls.py).
const AROUND_MIN = 24
const THROUGH_MIN = 3

interface Props {
  job: FemJob | null // the latest job of any kind
  results: FemJob | null // the latest analysis or sizing
  field: FemField | null
  quantity: FemQuantity
  caseId: string | null
  units: Units
  hasPart: boolean
  canSize: boolean
  setup: MeshSetupState | null
  mesh: FemMesh | null
  tab: FemTab
  onTab: (t: FemTab) => void
  showMesh: boolean
  onShowMesh: (b: boolean) => void
  bands: boolean
  onBands: (b: boolean) => void
  picking: PickKind | null
  onPick: (k: PickKind | null) => void
  active: ControlRef | null
  onActive: (c: ControlRef | null) => void
  onSetup: (s: Omit<MeshSetup, 'source'>) => void
  onPropose: (preset: string) => void
  onPlan: (instruction: string, maxMeshes: number) => void
  onMesh: () => void
  onRun: (convergence: boolean, bearing: FemBearing, peaks: boolean, peaksContact: boolean) => void
  onSize: (target: number, maxEvals: number) => void
  onStop: () => void
  onRevert: () => void
  onQuantity: (q: FemQuantity) => void
  onCase: (id: string) => void
  scale: number
  onScale: (s: number) => void
  onShowStatus: () => void
}

export function StressPanel(props: Props) {
  const { job, tab, onTab } = props
  const running = job?.status === 'running'
  return (
    <>
      <div className="ttl"><b>Stress analysis</b><span>{running ? `${job?.mode === 'plan' ? 'Claude meshing' : job?.mode === 'mesh' ? 'meshing' : job?.mode === 'size' ? 'sizing' : 'solving'}…` : props.results?.status === 'done' && props.results.cases.length ? `${props.results.cases.length} load case(s)` : 'not run'}</span></div>
      <div className="sp-row seg fem-tabs" role="tablist" aria-label="Stress analysis step">
        {(['mesh', 'results', 'size'] as FemTab[]).map((t, i) => (
          <button key={t} role="tab" aria-pressed={tab === t} aria-selected={tab === t} onClick={() => onTab(t)}>{i + 1} {t === 'mesh' ? 'Mesh' : t === 'results' ? 'Solve' : 'Size'}</button>
        ))}
      </div>
      {running && <button className="btn wide" onClick={props.onShowStatus}>Show solve status…</button>}
      {tab === 'mesh' ? <MeshTab {...props} /> : tab === 'results' ? <ResultsTab {...props} /> : <SizeTab {...props} />}
      {job && job.log.length > 0 && (
        <>
          <div className="sec">Log</div>
          <ol className="ailog">
            {job.log.slice(-40).map((l, i) => (
              <li key={i} className={`ai-${l.kind === 'error' ? 'error' : l.kind === 'result' ? 'finish' : l.kind === 'ai' ? 'ai' : 'read'}`}>
                <span className="ico" aria-hidden="true">{l.kind === 'error' ? '✖' : l.kind === 'result' ? '✓' : l.kind === 'mesh' ? '▦' : l.kind === 'statics' ? '⚖' : l.kind === 'ai' ? '✦' : '◇'}</span>
                <span className="tx">{l.text} <span className="aimeta">{l.t.toFixed(1)} s</span></span>
              </li>
            ))}
          </ol>
        </>
      )}
    </>
  )
}

/** A number input that commits on Enter or blur (not on every keystroke, which would re-save the setup each time). */
function NumField({ id, label, value, onCommit, disabled, step }: { id: string; label: string; value: number; onCommit: (v: number) => void; disabled?: boolean; step?: number }) {
  const [text, setText] = useState(String(value))
  useEffect(() => setText(String(value)), [value])
  const commit = () => {
    const v = Number(text)
    if (Number.isFinite(v) && v > 0 && v !== value) onCommit(v)
    else setText(String(value))
  }
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input id={id} type="number" step={step} value={text} disabled={disabled} onChange={(e) => setText(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} />
    </div>
  )
}

function MeshTab({ job, setup, mesh, units, hasPart, showMesh, onShowMesh, picking, onPick, active, onActive, onSetup, onPropose, onPlan, onMesh, onStop }: Props) {
  const [preset, setPreset] = useState('medium')
  const [instruction, setInstruction] = useState('')
  const [checks, setChecks] = useState(3)
  const [editPractices, setEditPractices] = useState(false)
  const [practices, setPractices] = useState('')
  const running = job?.status === 'running'
  const s = setup?.setup
  const L = (mm: number) => +toDisplay(mm, 'length', units).toFixed(units === 'in' ? 4 : 3)
  const M = (v: number) => fromDisplay(v, 'length', units)
  const u = units === 'in' ? 'in' : 'mm'
  const save = (patch: Partial<MeshSetup>) => { if (s) onSetup({ ...strip(s), ...patch }) }
  const busy = running || !hasPart
  const m = mesh?.metrics

  return (
    <>
      <p className="note">
        Quadratic tetrahedra. Set the mesh up here and check it before solving: the rules or Claude propose controls from the
        best practices, and you can edit any of them. Pick edges and faces on the part to add your own.
      </p>
      <div className="sec">Propose</div>
      <div className="row2">
        <div className="field">
          <label htmlFor="fem-preset">Best-practice rules at</label>
          <select id="fem-preset" value={preset} disabled={busy} onChange={(e) => setPreset(e.target.value)}>
            <option value="coarse">coarse density</option>
            <option value="medium">medium density</option>
            <option value="fine">fine density</option>
          </select>
        </div>
        <button className="btn" disabled={busy} onClick={() => onPropose(preset)}>Apply rules</button>
      </div>
      <div className="field">
        <label htmlFor="fem-plan">Ask Claude (optional instruction)</label>
        <textarea id="fem-plan" rows={2} value={instruction} disabled={busy} placeholder="e.g. keep it under 500k DOF; refine the IF-02 lug more"
          onChange={(e) => setInstruction(e.target.value)} />
      </div>
      <div className="sp-row">
        <button className="btn ai" disabled={busy} onClick={() => onPlan(instruction, checks)}>✦ Plan mesh with Claude</button>
        <select aria-label="Mesh checks Claude may run" value={checks} disabled={busy} onChange={(e) => setChecks(+e.target.value)}>
          {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n} check{n > 1 ? 's' : ''}</option>)}
        </select>
        {running && job?.mode === 'plan' && <button className="btn danger" onClick={onStop}>Stop</button>}
      </div>
      <button className="btn wide" onClick={() => { setPractices(s?.practices ?? ''); setEditPractices((v) => !v) }} aria-expanded={editPractices}>
        {editPractices ? 'Hide' : 'Edit'} best practices
      </button>
      {editPractices && s && (
        <div className="field">
          <label htmlFor="fem-practices">Best practices (the rules and Claude read these)</label>
          <textarea id="fem-practices" rows={10} value={practices} onChange={(e) => setPractices(e.target.value)} />
          <div className="sp-row">
            <button className="btn primary" disabled={practices === s.practices} onClick={() => save({ practices })}>Save</button>
            <button className="btn" onClick={() => setPractices(setup!.default_practices)}>Reset to defaults</button>
          </div>
        </div>
      )}

      {s && (
        <>
          <div className="sec">Controls <span className="aimeta">({s.source === 'none' ? 'none yet: solving applies the rules' : s.source})</span></div>
          {s.notes.length > 0 && <ul className="mesh-notes">{s.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
          <div className="row3">
            <NumField id="fem-size" label={`Global size (${u})`} value={L(s.size)} step={0.01} disabled={running} onCommit={(v) => save({ size: M(v) })} />
            <NumField id="fem-curv" label="Per circle" value={s.curvature} step={1} disabled={running} onCommit={(v) => save({ curvature: Math.round(v) })} />
            <NumField id="fem-min" label={`Min size (${u})`} value={L(s.min_size)} step={0.001} disabled={running} onCommit={(v) => save({ min_size: M(v) })} />
          </div>

          <ControlList title="Edge divisions" add="+ Edge" picking={picking === 'edge'} onAdd={() => onPick(picking === 'edge' ? null : 'edge')} disabled={running}
            hint="Click an edge on the part: exactly n elements along it.">
            {s.edges.map((e, i) => (
              <li key={i} className={active?.kind === 'edge' && active.index === i ? 'on' : ''} onMouseEnter={() => onActive({ kind: 'edge', index: i })} onMouseLeave={() => onActive(null)}>
                <span className="lbl">{e.label || `edge ${e.edge}`}{setup!.resolved.edges[i] == null && <span className="wn" title="This edge is not on the current part (it has changed); the control is skipped"> ⚠ not found</span>}</span>
                <input aria-label="Divisions" type="number" min={1} max={500} value={e.n} disabled={running}
                  onChange={(ev) => { const n = Math.max(1, Math.min(500, Math.round(+ev.target.value || 1))); save({ edges: s.edges.map((x, j) => (j === i ? { ...x, n } : x)) }) }} />
                <button className="x" aria-label="Remove" disabled={running} onClick={() => save({ edges: s.edges.filter((_, j) => j !== i) })}>×</button>
              </li>
            ))}
          </ControlList>

          <ControlList title="Mapped faces" add="+ Face" picking={picking === 'face'} onAdd={() => onPick(picking === 'face' ? null : 'face')} disabled={running}
            hint="Click a face with 3 or 4 edges and no holes: a structured grid of triangles on it.">
            {s.faces.map((f, i) => (
              <li key={i} className={active?.kind === 'face' && active.index === i ? 'on' : ''} onMouseEnter={() => onActive({ kind: 'face', index: i })} onMouseLeave={() => onActive(null)}>
                <span className="lbl">{f.label || `face ${f.face}`}{setup!.resolved.faces[i] == null && <span className="wn"> ⚠ not found</span>}</span>
                <button className="x" aria-label="Remove" disabled={running} onClick={() => save({ faces: s.faces.filter((_, j) => j !== i) })}>×</button>
              </li>
            ))}
          </ControlList>

          <ControlList title="Spheres of influence" add="+ Sphere" picking={picking === 'sphere'} onAdd={() => onPick(picking === 'sphere' ? null : 'sphere')} disabled={running}
            hint="Click the part where the sphere should be centred: finer elements inside it.">
            {s.spheres.map((sp, i) => (
              <li key={i} className={`sphere${active?.kind === 'sphere' && active.index === i ? ' on' : ''}`} onMouseEnter={() => onActive({ kind: 'sphere', index: i })} onMouseLeave={() => onActive(null)}>
                <span className="lbl">{sp.label || `sphere ${i + 1}`}</span>
                <label>R <input aria-label="Radius" type="number" step={0.01} value={L(sp.radius)} disabled={running}
                  onChange={(ev) => { const v = +ev.target.value; if (v > 0) save({ spheres: s.spheres.map((x, j) => (j === i ? { ...x, radius: M(v) } : x)) }) }} /></label>
                <label>size <input aria-label="Element size" type="number" step={0.001} value={L(sp.size)} disabled={running}
                  onChange={(ev) => { const v = +ev.target.value; if (v > 0) save({ spheres: s.spheres.map((x, j) => (j === i ? { ...x, size: M(v) } : x)) }) }} /></label>
                <button className="x" aria-label="Remove" disabled={running} onClick={() => save({ spheres: s.spheres.filter((_, j) => j !== i) })}>×</button>
              </li>
            ))}
          </ControlList>
        </>
      )}

      <div className="sec">Check</div>
      <div className="sp-row">
        <button className="btn primary" disabled={busy} onClick={onMesh}>{running && job?.mode === 'mesh' ? 'Meshing…' : 'Generate mesh'}</button>
        <label className="sp-check" htmlFor="fem-showmesh"><input id="fem-showmesh" type="checkbox" checked={showMesh} onChange={(e) => onShowMesh(e.target.checked)} /> Show mesh</label>
      </div>
      {mesh?.stale && <p className="note wn">⚠ The controls or the part have changed since this mesh. Generate it again (solving does it anyway).</p>}
      {m && (
        <>
          <dl className="kv">
            <dt>Elements</dt><dd>{m.tets.toLocaleString()} tet10, {m.nodes.toLocaleString()} nodes</dd>
            <dt>DOF</dt><dd>{m.dof.toLocaleString()} <span className={m.gb > 12 ? 'no' : m.gb > 7 ? 'wn' : 'note'}>≈ {m.gb.toFixed(1)} GB to solve</span></dd>
            <dt>Quality</dt><dd>min <span className={m.quality.min < 0.1 ? 'no' : m.quality.min < 0.2 ? 'wn' : 'ok'}>{m.quality.min.toFixed(2)}</span>, mean {m.quality.mean.toFixed(2)}, {m.quality.below_0_2} below 0.2</dd>
          </dl>
          <QualityBars hist={m.quality.histogram} />
          <table className="holes topo">
            <thead><tr><th>Bore</th><th>Around</th><th>Through wall</th></tr></thead>
            <tbody>
              {m.bores.map((b) => (
                <tr key={b.id}>
                  <td>{b.id}</td>
                  <td className={b.around >= AROUND_MIN ? 'ok' : 'wn'}>{b.around} {b.around >= AROUND_MIN ? '✓' : `(< ${AROUND_MIN})`}</td>
                  <td className={(b.through ?? 0) >= THROUGH_MIN ? 'ok' : 'wn'}>{b.through ?? '?'} over {len(b.thickness, units)} {u} {(b.through ?? 0) >= THROUGH_MIN ? '✓' : `(< ${THROUGH_MIN})`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {m.notes.map((n, i) => <p key={i} className="note wn">⚠ {n}</p>)}
          <p className="note">Quality is the scaled Jacobian of each element (1 ideal, below 0.2 poor). Ticks mark the best practices: at least {AROUND_MIN} elements round each bore and {THROUGH_MIN} through its wall.</p>
        </>
      )}
    </>
  )
}

function ControlList({ title, add, picking, onAdd, disabled, hint, children }: { title: string; add: string; picking: boolean; onAdd: () => void; disabled: boolean; hint: string; children: React.ReactNode[] }) {
  return (
    <div className="ctl-list">
      <div className="ctl-head">
        <span>{title} <span className="aimeta">{children.length}</span></span>
        <button className="btn" aria-pressed={picking} disabled={disabled} onClick={onAdd}>{picking ? 'Cancel pick' : add}</button>
      </div>
      {picking && <p className="note pick-hint">{hint} Esc cancels.</p>}
      {children.length > 0 && <ul>{children}</ul>}
    </div>
  )
}

function QualityBars({ hist }: { hist: number[] }) {
  const max = Math.max(1, ...hist)
  return (
    <div className="qbars" role="img" aria-label={`Element quality histogram: ${hist.join(', ')} elements per tenth from 0 to 1`}>
      {hist.map((h, i) => (
        <div key={i} className="qb" title={`${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}: ${h.toLocaleString()} elements`}>
          <div style={{ height: `${Math.max(h > 0 ? 3 : 0, (100 * h) / max)}%`, background: i < 2 ? 'var(--bad)' : i < 3 ? 'var(--warn, #a86a10)' : 'var(--good, #23824f)' }} />
        </div>
      ))}
      <span className="qa">0</span><span className="qz">1</span>
    </div>
  )
}

function ResultsTab({ job: current, results: job, field, quantity, caseId, units, hasPart, mesh, onRun, onQuantity, onCase, scale, onScale, showMesh, onShowMesh, bands, onBands }: Props) {
  const [conv, setConv] = useState(false)
  const [bearing, setBearing] = useState<FemBearing>('cosine')
  const [peaks, setPeaks] = useState(false)
  const [peaksContact, setPeaksContact] = useState(false)
  const running = current?.status === 'running'
  const stress = (mpa: number) => (units === 'in' ? `${(mpa / KSI).toFixed(1)} ksi` : `${mpa.toFixed(0)} MPa`)
  const disp = (mm: number) => (units === 'in' ? `${(mm / 25.4).toFixed(4)} in` : `${mm.toFixed(3)} mm`)
  const ms = (v: number) => <span className={v < 0 ? 'no' : v < 0.15 ? 'wn' : 'ok'}>{v >= 0 ? '+' : ''}{v.toFixed(2)}</span>
  const c = job?.cases.find((x) => x.id === caseId) ?? job?.cases[0]
  const mat = job?.stats.material
  const fmt = (v: number) => (field?.quantity === 'u' ? disp(v) : stress(v))

  return (
    <>
      <div className="sp-row seg" role="group" aria-label="Bearing model">
        <button aria-pressed={bearing === 'cosine'} disabled={running} onClick={() => setBearing('cosine')}>Cosine bearing (fast)</button>
        <button aria-pressed={bearing === 'contact'} disabled={running} onClick={() => setBearing('contact')}>Pin contact</button>
      </div>
      <p className="note">
        {bearing === 'cosine'
          ? 'Puts each pin\'s force (from statics) on its bore as cosine bearing pressure: one linear solve, about a minute.'
          : 'Models a steel pin in each bore (0.0005 in diametral clearance), held square by its fork, with frictionless contact: where each pin bears comes out of the solve, pin bending included. Several passes, a few minutes. The cosine model is solved on the same mesh too, and each region\'s ratio calibrates it.'}
        {' '}Margins: limit load against Fty, ultimate against Ftu, on von Mises.
      </p>
      <p className="note">
        Mesh: {mesh ? `${mesh.metrics.dof.toLocaleString()} DOF${mesh.stale ? ' (out of date: it is re-made from the controls when you solve)' : ' (checked)'}` : 'not generated yet: solving makes it from the controls'}.
      </p>
      <label className="sp-check" htmlFor="fem-conv" title="Solve again with every element size ×1.5 and report how much each region's peak moves">
        <input id="fem-conv" type="checkbox" checked={conv} disabled={running} onChange={(e) => setConv(e.target.checked)} /> Convergence check (re-solve on a coarser mesh)
      </label>
      <label className="sp-check" htmlFor="fem-peaks" title="Refine the mesh locally at every hot spot within 10% of the peak and solve again, until each peak moves less than 5%">
        <input id="fem-peaks" type="checkbox" checked={peaks} disabled={running} onChange={(e) => setPeaks(e.target.checked)} /> Converge the peaks (local refinement until each moves &lt; 5%)
      </label>
      {peaks && bearing === 'contact' && (
        <label className="sp-check sub" htmlFor="fem-peaks-contact" title="Otherwise the rounds solve the cosine model and each spot carries its contact / cosine factor from this run">
          <input id="fem-peaks-contact" type="checkbox" checked={peaksContact} disabled={running} onChange={(e) => setPeaksContact(e.target.checked)} /> Pin contact in every round (slow: several minutes a round)
        </label>
      )}
      {peaks && (
        <p className="note">
          Each round adds a sphere of finer elements at every spot still moving (about 1.5× finer each time) and solves again, up to 4 rounds.
          {bearing === 'contact' && !peaksContact ? ' The rounds solve the cosine model; each spot\'s contact / cosine factor from this run carries the result back.' : ''}
          {' '}A peak that keeps climbing is reported as a singularity (a sharp corner wants a fillet, not a finer mesh).
        </p>
      )}
      <button className="btn primary" disabled={!hasPart || running} onClick={() => onRun(conv, bearing, peaks, peaksContact)}>
        {running && job?.mode === 'analyse' ? 'Solving…' : 'Run stress analysis'}
      </button>
      {!hasPart && <p className="note">Make a part first (topology → parametric model, or a reference bracket).</p>}
      {job?.stale && <p className="note wn">⚠ The part has changed since this analysis. Run it again.</p>}
      {c && (
        <>
          <div className="sec">Results</div>
          {job && job.cases.length > 1 && (
            <div className="field">
              <label htmlFor="fem-case">Load case</label>
              <select id="fem-case" value={c.id} onChange={(e) => onCase(e.target.value)}>
                {job.cases.map((x) => <option key={x.id} value={x.id}>{x.id} {x.label}</option>)}
              </select>
            </div>
          )}
          <dl className="kv">
            <dt>Peak von Mises</dt><dd>{stress(c.vm_max)} ({c.category}) at {c.regions[0]?.name}</dd>
            <dt>Peak principal</dt><dd>{stress(c.p1_max)}</dd>
            <dt>MS yield</dt><dd>{ms(c.ms_yield)} <span className="note">limit vs Fty {mat ? stress(mat.fty) : ''}</span></dd>
            <dt>MS ultimate</dt><dd>{ms(c.ms_ultimate)} <span className="note">{c.category === 'limit' ? `${c.factor} × limit` : 'ultimate'} vs Ftu {mat ? stress(mat.ftu) : ''}</span></dd>
            {c.peaks?.governing && (
              <><dt>Converged peak</dt><dd>{stress(c.peaks.governing.vm)} at {c.peaks.governing.region}, MS ult {ms(c.peaks.governing.ms_ultimate)}
                {c.peaks.governing.status !== 'converged' && <span className="note wn"> (not converged)</span>}</dd></>
            )}
            <dt>Bearing model</dt><dd>{c.bearing === 'contact' ? `Pin contact (${c.contact?.passes ?? '?'} passes)` : 'Cosine bearing'}</dd>
            <dt>Max displacement</dt><dd>{disp(c.u_max)}{c.bearing === 'contact' ? ' (from the forks)' : ' (from the hold points)'}</dd>
            <dt>Pin forces</dt><dd>{c.pins.map((p) => `${p.id} ${p.role} ${(Math.hypot(...p.force) / 4.4482216).toFixed(0)} lbf`).join(' · ')}</dd>
          </dl>
          <table className="holes topo">
            <thead><tr><th>Region</th><th>von Mises</th><th>Principal</th><th>MS ult</th>{c.convergence && <th>vs coarser</th>}{c.calibration && Object.keys(c.calibration).length > 0 && <th title="Contact peak / cosine-model peak on the same mesh">vs cosine</th>}</tr></thead>
            <tbody>
              {c.regions.map((g) => (
                <tr key={g.name}>
                  <td>{g.name}</td><td>{stress(g.vm)}</td><td>{stress(g.p1)}</td><td>{ms(g.ms_ultimate)}</td>
                  {c.convergence && <td>{c.convergence.regions[g.name] == null ? '' : `${(100 * c.convergence.regions[g.name]!).toFixed(1)}%`}</td>}
                  {c.calibration && Object.keys(c.calibration).length > 0 && <td className={(c.calibration[g.name] ?? 1) > 1.15 ? 'wn' : ''}>{c.calibration[g.name] ? `×${c.calibration[g.name].toFixed(2)}` : ''}</td>}
                </tr>
              ))}
            </tbody>
          </table>
          {c.peaks && c.peaks.spots.length > 0 && <PeakTable peaks={c.peaks} stress={stress} ms={ms} />}
          {c.contact?.pins && (
            <table className="holes topo">
              <thead><tr><th>Pin</th><th>Peak bearing</th><th>Pin von Mises</th><th>Force (contact vs statics)</th></tr></thead>
              <tbody>
                {Object.entries(c.contact.pins).map(([id, v]) => (
                  <tr key={id}><td>{id}</td><td>{stress(v.p_max)}</td><td>{stress(v.pin_vm)}</td>
                    <td>{(Math.hypot(...v.force) / 4.4482216).toFixed(0)} vs {(Math.hypot(...v.statics) / 4.4482216).toFixed(0)} lbf</td></tr>
                ))}
              </tbody>
            </table>
          )}
          {c.residual > 0.005 * Math.max(...c.pins.map((p) => Math.hypot(...p.force))) && (
            <p className="note wn">
              ⚠ The rigid-body hold reacts {(c.residual / 4.4482216).toFixed(0)} lbf: it should carry about nothing. The pin forces
              balance in the plane, but bearing surfaces at different heights (one-sided spot faces) leave an out-of-plane couple that
              only the hold points resist here, where the forks would react it. Treat displacements and nearby stresses with care.
            </p>
          )}
          {mat && <p className="note">Allowables: {mat.source}. Check them against your handbook; override Ftu / Fty in Design rules.</p>}
          <div className="sec">Contours</div>
          <div className="sp-row seg" role="group" aria-label="Contour shown">
            {(Object.keys(QTY) as FemQuantity[]).map((q) => (
              <button key={q} aria-pressed={quantity === q} onClick={() => onQuantity(q)}>{q === 'vm' ? 'von Mises' : q === 'p1' ? 'Principal' : 'Displacement'}</button>
            ))}
          </div>
          <div className="sp-row seg" role="group" aria-label="Contour style">
            <button aria-pressed={!bands} onClick={() => onBands(false)}>Gradient</button>
            <button aria-pressed={bands} onClick={() => onBands(true)}>{BANDS} bands</button>
          </div>
          <label className="sp-check" htmlFor="fem-showmesh-r"><input id="fem-showmesh-r" type="checkbox" checked={showMesh} onChange={(e) => onShowMesh(e.target.checked)} /> Show mesh</label>
          {field && (
            <div className="legend-bar">
              {bands ? <BandLegend lo={field.min} hi={field.max} fmt={fmt} /> : (
                <>
                  <div className="bar" style={{ background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => rgb(turbo(t))).join(', ')})` }} />
                  <div className="ends"><span>{fmt(field.min)}</span><span>{QTY[field.quantity]}</span><span>{fmt(field.max)}</span></div>
                </>
              )}
              <div className="field">
                <label htmlFor="fem-scale">Deformed shape: {scale > 0 ? `× ${Math.round(scale)}` : 'off'}</label>
                <input id="fem-scale" type="range" min={0} max={autoScale(field) * 2} step={autoScale(field) / 20} value={scale}
                  onChange={(e) => onScale(+e.target.value)} />
                <span className="note">Exaggerated displacements: check the part bends the way the supports and loads say it should.</span>
              </div>
              <p className="note">● marks the peak. Mating parts are hidden while this panel is open. Contours are on the outer surface; use Section to look inside.</p>
            </div>
          )}
        </>
      )}
    </>
  )
}

/** One swatch per band with the value at every band edge (as a stress engineer reads a banded plot). */
function BandLegend({ lo, hi, fmt }: { lo: number; hi: number; fmt: (v: number) => string }) {
  const edges = Array.from({ length: BANDS + 1 }, (_, i) => lo + ((hi - lo) * i) / BANDS)
  return (
    <div className="band-legend" role="table" aria-label="Contour bands">
      {Array.from({ length: BANDS }, (_, i) => BANDS - 1 - i).map((i) => (
        <div key={i} className="band-row" role="row">
          <span className="sw" style={{ background: rgb(bandColor(i, BANDS)) }} />
          <span role="cell">{fmt(edges[i])} – {fmt(edges[i + 1])}</span>
        </div>
      ))}
    </div>
  )
}

function SizeTab({ job: current, results: job, units, hasPart, canSize, onSize, onStop, onRevert }: Props) {
  const [target, setTarget] = useState('0.10')
  const [evals, setEvals] = useState(20)
  const running = current?.status === 'running'
  const lb = (kg: number) => (units === 'in' ? `${(kg / 0.45359237).toFixed(3)} lb` : `${kg.toFixed(3)} kg`)
  const sz = job?.mode === 'size' ? job.sizing : null
  return (
    <>
      <p className="note">
        Finds the lightest version of the drafted part that meets the margin: thins the web and rim, grows a lug that governs
        (never below the e/D rule), with a coarse analysis per trial (best-practice rules mesh). Keeps the best and lets you revert.
      </p>
      <div className="row2">
        <div className="field">
          <label htmlFor="fem-target">Target MS ultimate</label>
          <input id="fem-target" value={target} disabled={running} onChange={(e) => setTarget(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="fem-evals">Trials (≈20 s each)</label>
          <select id="fem-evals" value={evals} disabled={running} onChange={(e) => setEvals(+e.target.value)}>
            {[8, 12, 20, 30].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      </div>
      <div className="sp-row">
        <button className="btn" disabled={!canSize || running || !Number.isFinite(Number(target))} onClick={() => onSize(Number(target), evals)}
          title={canSize ? '' : 'Sizing works on a part drafted with AI (it has a recipe)'}>Size the part</button>
        {running && job?.mode === 'size' && <button className="btn danger" onClick={onStop}>Stop</button>}
        {sz?.can_revert && !running && <button className="btn" onClick={onRevert}>Revert sizing</button>}
      </div>
      {!canSize && hasPart && <p className="note">Sizing needs a part drafted with AI (Topology → Draft with AI).</p>}
      {sz && sz.after_mass != null && (
        <p className={`note ${sz.feasible ? 'ok' : 'wn'}`}>
          {lb(sz.before_mass)} → {lb(sz.after_mass)} in {sz.trials} trials{sz.feasible ? '' : ' (target not met; kept the closest)'}
        </p>
      )}
    </>
  )
}

function strip(s: MeshSetup): Omit<MeshSetup, 'source'> {
  const { source: _source, ...rest } = s
  return rest
}

/** A magnification that makes the largest displacement about 4% of the part's size. */
function autoScale(f: FemField): number {
  let umax = 0
  for (let i = 0; i < f.displacement.length; i += 3) umax = Math.max(umax, Math.hypot(f.displacement[i], f.displacement[i + 1], f.displacement[i + 2]))
  let lo = Infinity, hi = -Infinity
  for (let i = 0; i < f.positions.length; i += 3) { lo = Math.min(lo, f.positions[i]); hi = Math.max(hi, f.positions[i]) }
  return umax > 0 ? Math.max(1, (0.04 * (hi - lo)) / umax) : 1
}

const rgb = ([r, g, b]: [number, number, number]) => `rgb(${Math.round(255 * r)},${Math.round(255 * g)},${Math.round(255 * b)})`

const PEAK_STATUS: Record<FemPeakSpot['status'], string> = { converged: 'converged', singular: 'singularity', limit: 'not converged', converging: 'refining…' }

/** Each hot spot's peak mesh by mesh, its status and margin. */
export function PeakTable({ peaks, stress, ms }: { peaks: FemPeaks; stress: (mpa: number) => string; ms?: (v: number) => React.ReactNode }) {
  const sing = peaks.spots.some((s) => s.status === 'singular')
  return (
    <>
      <table className="holes topo">
        <thead><tr><th>Hot spot</th><th title={peaks.factored ? 'Cosine-model peak per mesh, coarsest first' : 'Peak per mesh, coarsest first'}>By mesh ({stress(0).split(' ')[1]})</th><th>Status</th>
          <th title={peaks.factored ? 'Finest mesh times the contact / cosine factor at this spot' : 'Finest mesh'}>Peak</th>{ms && <th>MS ult</th>}</tr></thead>
        <tbody>
          {peaks.spots.map((s) => (
            <tr key={s.id}>
              <td>{s.region || s.id}</td>
              <td className="nowrap-list">{s.history.map((h, i) => <span key={i}>{i > 0 && '→'}<span className="nw">{stress(h.vm).split(' ')[0]}</span>{' '}</span>)}</td>
              <td className={s.status === 'converged' ? 'ok' : s.status === 'singular' ? 'no' : 'wn'}
                title={`Element ${s.history[0].h.toFixed(2)} → ${s.history[s.history.length - 1].h.toFixed(2)} mm${s.change != null ? `; last change ${(100 * s.change).toFixed(1)}%` : ''}`}>{PEAK_STATUS[s.status]}</td>
              <td>{stress(s.vm)}{peaks.factored && <div className="note">×{s.factor.toFixed(2)} contact</div>}</td>
              {ms && <td>{s.ms_ultimate != null ? ms(s.ms_ultimate) : ''}</td>}
            </tr>
          ))}
        </tbody>
      </table>
      {sing && <p className="note wn">⚠ A singularity: the peak keeps climbing as the mesh refines (a sharp inside corner, a point load or an edge of contact). No mesh gives a true value there; add a fillet or check the detail by hand.</p>}
    </>
  )
}
