import { useEffect, useState } from 'react'
import type { InterpretJob, Machining, TopoStatus, TopoStep, TopoSummary } from './api'
import { SKETCH_COLORS } from './viewer/Viewer'
import { fromDisplay, len, toDisplay, UNIT_LABEL, type Units } from './units'

export const BUDGETS = [20_000, 40_000, 60_000, 100_000, 150_000, 250_000]

export interface TopoView {
  step: number | 'live'
  cutoff: number
}

interface Props {
  status: TopoStatus | null
  summary: TopoSummary | null
  setupError: string | null
  setupLoading: boolean
  budget: number
  showSetup: boolean
  view: TopoView
  shownMass: number | null
  massTarget: number | null
  units: Units
  onBudget: (n: number) => void
  onShowSetup: (on: boolean) => void
  onView: (v: TopoView) => void
  machining: Machining | null
  onMachining: (m: Machining | null) => void
  onRun: (mass: number | null, sweep: boolean) => void
  onCancel: () => void
  interp: InterpretJob | null
  hasDesign: boolean
  showPart: boolean
  onShowPart: (on: boolean) => void
  showTrace: boolean
  onShowTrace: (on: boolean) => void
  onInterpret: (step: number, mode?: 'rules' | 'ai', instruction?: string, maxBuilds?: number, stress?: boolean) => void
  onInterpretCancel: () => void
  aiDraft: { summary: string; questions: string[] } | null
}

export function TopoPanel({ status, summary, setupError, setupLoading, budget, showSetup, view, shownMass, massTarget, units, machining, onMachining, onBudget, onShowSetup, onView, onRun, onCancel, interp, hasDesign, showPart, onShowPart, showTrace, onShowTrace, onInterpret, onInterpretCancel, aiDraft }: Props) {
  // The project's mass target by default; editable here for a one-off run without changing the rules.
  const show = (kg: number | null) => (kg == null ? '' : toDisplay(kg, 'mass', units).toFixed(3))
  const [massText, setMassText] = useState(show(massTarget))
  useEffect(() => setMassText(show(massTarget)), [massTarget, units]) // eslint-disable-line react-hooks/exhaustive-deps
  const massKg = massText.trim() && Number(massText) > 0 ? fromDisplay(Number(massText), 'mass', units) : null
  const fraction = massKg != null && summary ? massKg / summary.full_mass : null
  const job = status?.job ?? null
  const running = job?.status === 'running'
  const result = status?.result ?? null
  const mass = (kg: number) => `${toDisplay(kg, 'mass', units).toFixed(3)} ${UNIT_LABEL[units].mass}`
  const pct = (f: number) => `${(f * 100).toFixed(f < 0.1 ? 1 : 0)}%`
  const iterSecs = job && job.history.length ? job.history.slice(-10).reduce((a, h) => a + h.s, 0) / Math.min(10, job.history.length) : null
  const steps = result?.steps ?? []
  const stepIdx = typeof view.step === 'number' ? Math.min(view.step, steps.length - 1) : -1

  return (
    <>
      <div className="ttl">
        <b>Topology optimisation</b>
        <span>{running ? 'running' : result ? `${steps.length} results` : 'not run'}</span>
      </div>
      <p className="note">
        Maximises stiffness for a given mass inside the design space, holding material at the interfaces and out of keep-outs.
        It optimises for one mass; use the density cutoff to look around the result, or sweep to see the layouts a step heavier and lighter.
      </p>

      <div className="sec">Mesh</div>
      <div className="field">
        <label htmlFor="topo-budget">Element budget</label>
        <select id="topo-budget" value={budget} disabled={running} onChange={(e) => onBudget(+e.target.value)}>
          {BUDGETS.map((b) => <option key={b} value={b}>{(b / 1000).toFixed(0)}k elements</option>)}
        </select>
      </div>
      {setupError ? (
        <p className="errbox">{setupError}</p>
      ) : summary ? (
        <>
          <dl className="kv">
            <dt>Element size</dt><dd>{len(summary.h, units, units === 'in' ? 4 : 2)} {units} cubes</dd>
            <dt>Grid</dt><dd>{summary.grid.join(' × ')}</dd>
            <dt>Elements</dt><dd>{summary.elements.toLocaleString()} of {summary.budget.toLocaleString()}</dd>
            <dt>At interfaces</dt><dd>{summary.solid.toLocaleString()} held solid</dd>
            <dt>Left out</dt><dd>{summary.void.toLocaleString()} (keep-outs, mating parts, bolt and pin paths)</dd>
            <dt>All solid</dt><dd>{mass(summary.full_mass)}</dd>
            <dt>Supports</dt><dd>{summary.supports.map((s) => `${s.id} ${s.kind}${s.along ? ` (${s.along})` : ''}`).join(', ')}</dd>
            <dt>Loads</dt><dd>{summary.loads.map((l) => `${l.id} at ${l.target}`).join(', ')}</dd>
          </dl>
          {summary.notes.map((n) => <p key={n} className="note wn">⚠ {n}</p>)}
          <label className="sp-check" htmlFor="topo-show-setup">
            <input id="topo-show-setup" type="checkbox" checked={showSetup} onChange={(e) => onShowSetup(e.target.checked)} /> Show set-up in viewport
          </label>
          {showSetup && (
            <p className="note legend">
              <span className="dot" style={{ background: '#2f9e8f' }} /> held solid <span className="dot" style={{ background: '#e0782a' }} /> fixed / pin
              <span className="dot" style={{ background: '#d9b52a' }} /> normal / link <span className="dot" style={{ background: '#8a4fd6' }} /> load
            </p>
          )}
        </>
      ) : (
        <p className="note">{setupLoading ? 'Meshing…' : ''}</p>
      )}

      <div className="sec">Machining</div>
      <div className="field">
        <label htmlFor="topo-method">Constrain the shape to</label>
        <select id="topo-method" value={machining ? '4axis' : 'raw'} disabled={running}
          onChange={(e) => onMachining(e.target.value === 'raw' ? null : DEFAULT_MACHINING)}>
          <option value="raw">Nothing (raw result)</option>
          <option value="4axis">2.5D on a 4-axis mill</option>
        </select>
      </div>
      {machining && summary && (
        <MachiningFields m={machining} grid={summary.grid} units={units} disabled={running} onChange={onMachining} />
      )}

      <div className="sec">Run</div>
      <div className="field">
        <label htmlFor="topo-mass">Mass target ({UNIT_LABEL[units].mass}){fraction != null && ` · ${pct(fraction)} of the space`}</label>
        <input id="topo-mass" type="number" min={0} step={units === 'in' ? 0.01 : 0.005} value={massText} disabled={running}
          placeholder="set one in Design rules" onChange={(e) => setMassText(e.target.value)} />
      </div>
      <div className="sp-row">
        {running ? (
          <button className="btn danger" onClick={onCancel}>Stop</button>
        ) : (
          <>
            <button className="btn primary" disabled={!summary || !!setupError || massKg == null} onClick={() => onRun(massKg, false)}
              title="One optimisation for exactly this mass">Run at mass target</button>
            <button className="btn" disabled={!summary || !!setupError || massKg == null || !!machining} onClick={() => onRun(massKg, true)}
              title={machining ? 'Sweeps are for raw runs; a machinable run already shows the raw result beside it' : 'Also 1.5× and 0.65× the target, each starting from the last, to see how the load paths change'}>Sweep ×1.5 / ×0.65</button>
          </>
        )}
      </div>
      {machining && <p className="note">Runs the raw optimum first, then fits the 2.5D layers to it and re-optimises, so you see both at the same mass.</p>}
      {iterSecs != null && <p className="note">{iterSecs.toFixed(1)} s per iteration</p>}

      {job && running && (
        <>
          <div className="sec">
            Step {job.step + 1} of {job.fractions.length || '…'}
            {job.labels?.[job.step] && job.labels[job.step] !== 'raw' ? ` · ${job.labels[job.step]}` : job.fractions[job.step] != null ? ` · raw, ${pct(job.fractions[job.step])} of the space` : ''} · iteration {job.iteration}
            <span className="spin inline" aria-label="working" />
          </div>
          <Convergence history={job.history.filter((h) => h.step === job.step)} />
          <label className="sp-check" htmlFor="topo-live">
            <input id="topo-live" type="checkbox" checked={view.step === 'live'} disabled={!steps.length}
              onChange={(e) => onView({ ...view, step: e.target.checked ? 'live' : steps.length - 1 })} /> Watch it live
          </label>
        </>
      )}
      {job && !running && job.status !== 'done' && (
        <p className={job.status === 'failed' ? 'errbox' : 'note'}>Last run {job.status}{job.error ? `: ${job.error}` : ''}</p>
      )}

      {steps.length > 0 && (
        <>
          <div className="sec">Results</div>
          {status?.stale && <p className="note wn">⚠ The intent has changed since this run. Run again to update it.</p>}
          {result?.machining && steps.length === 2 ? (
            <MachinedToggle steps={steps} stepIdx={stepIdx} onStep={(i) => onView({ ...view, step: i })} mass={mass} />
          ) : steps.length > 1 ? (
            <div className="field">
              <label htmlFor="topo-step">
                Mass target{stepIdx >= 0 ? `: ${mass(steps[stepIdx].mass)} (${pct(steps[stepIdx].fraction)})` : ': live'}
                {stepIdx >= 0 && stepIdx === result?.target_step && ' ★ your target'}
              </label>
              <input id="topo-step" type="range" min={0} max={steps.length - 1} step={1} value={Math.max(0, stepIdx)}
                onChange={(e) => onView({ ...view, step: +e.target.value })} />
              <div className="ticks">
                <span>{mass(steps[0].mass)}</span>
                <span>{mass(steps[steps.length - 1].mass)}</span>
              </div>
            </div>
          ) : (
            <p className="note">Optimised for {mass(steps[0].mass)} ({pct(steps[0].fraction)} of the space).</p>
          )}
          {result?.machining && stepIdx === 1 ? (
            <p className="note">Shown as the machined part: profiles and flat-floored pockets, walls parallel to the cutter (stair-stepped at the element size until it becomes features).</p>
          ) : <>
          <div className="field">
            <label htmlFor="topo-cutoff">Density cutoff: {view.cutoff.toFixed(2)}{shownMass != null && ` · ${mass(shownMass)} shown`}</label>
            <input id="topo-cutoff" type="range" min={0.05} max={0.95} step={0.05} value={view.cutoff}
              onChange={(e) => onView({ ...view, cutoff: +e.target.value })} />
          </div>
          <p className="note">
            The cutoff hides elements below that density: raise it to see the main load paths, lower it to see where the optimiser was
            undecided. Small trims stay close to optimal; for a much lighter part, run at that mass.
          </p>
          </>}
          {!result?.machining && <table className="holes topo">
            <thead><tr><th>Step</th><th>Mass</th><th>Space</th><th>Iter.</th></tr></thead>
            <tbody>
              {steps.map((s) => (
                <tr key={s.index} className={s.index === stepIdx ? 'sel' : ''} onClick={() => onView({ ...view, step: s.index })}>
                  <td>{s.index + 1}{s.index === result?.target_step ? ' ★' : ''}</td>
                  <td>{mass(s.mass)}</td>
                  <td>{pct(s.fraction)}</td>
                  <td>{s.iterations}{s.converged ? '' : '+'}</td>
                </tr>
              ))}
            </tbody>
          </table>}
          {steps.some((st) => !st.converged) && <p className="note">Iter. marked + stopped at the iteration limit before settling.</p>}
          <ParametricModel step={stepIdx} steps={steps} stale={!!status?.stale} busy={running} job={interp} hasDesign={hasDesign}
            showPart={showPart} onShowPart={onShowPart} showTrace={showTrace} onShowTrace={onShowTrace} onStart={onInterpret} onStop={onInterpretCancel} mass={mass} aiDraft={aiDraft} />
        </>
      )}
    </>
  )
}

/** Normalised compliance (1 = the first iteration) against iteration, for the step in progress. */
function Convergence({ history }: { history: { it: number; c: number }[] }) {
  if (history.length < 2) return null
  const w = 240
  const h = 56
  const cs = history.map((p) => p.c)
  const lo = Math.min(...cs)
  const hi = Math.max(...cs)
  const x = (i: number) => (i / (history.length - 1)) * (w - 4) + 2
  const y = (c: number) => h - 4 - ((c - lo) / (hi - lo || 1)) * (h - 8)
  const d = history.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.c).toFixed(1)}`).join(' ')
  return (
    <figure className="conv">
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} role="img" aria-label="Compliance by iteration">
        <path d={d} fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
      <figcaption>Compliance (lower is stiffer) · {cs[cs.length - 1].toPrecision(3)} of the start</figcaption>
    </figure>
  )
}

export const DEFAULT_MACHINING: Machining = { axis: 'auto', depths: 3, free_end: 'none', tool_diameter: 6.35 }

function MachiningFields({ m, grid, units, disabled, onChange }: { m: Machining; grid: [number, number, number]; units: Units; disabled: boolean; onChange: (m: Machining) => void }) {
  const longest = 'XYZ'[grid.indexOf(Math.max(...grid))]
  const axis = m.axis === 'auto' ? longest : m.axis.toUpperCase()
  const sides = 'XYZ'.split('').filter((a) => a !== axis)
  const faces = [...sides.flatMap((a) => [`+${a}`, `−${a}`]), ...(m.free_end === 'none' ? [] : [`${m.free_end === '+' ? '+' : '−'}${axis}`])].join(', ')
  const toolShown = toDisplay(m.tool_diameter, 'length', units).toFixed(units === 'in' ? 3 : 2)
  const [tool, setTool] = useState(toolShown)
  useEffect(() => setTool(toolShown), [toolShown])
  const commitTool = () => {
    const v = Number(tool)
    if (Number.isFinite(v) && v > 0) onChange({ ...m, tool_diameter: fromDisplay(v, 'length', units) })
    else setTool(toolShown)
  }
  return (
    <>
      <div className="row2">
        <div className="field">
          <label htmlFor="mach-axis">Rotary axis</label>
          <select id="mach-axis" value={m.axis} disabled={disabled} onChange={(e) => onChange({ ...m, axis: e.target.value as Machining['axis'] })}>
            <option value="auto">Long axis ({longest})</option>
            <option value="x">X</option>
            <option value="y">Y</option>
            <option value="z">Z</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="mach-depths">Floor depths per face</label>
          <select id="mach-depths" value={m.depths} disabled={disabled} onChange={(e) => onChange({ ...m, depths: +e.target.value })}>
            {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n === 1 ? '1 (profiles only)' : n}</option>)}
          </select>
        </div>
      </div>
      <div className="row2">
        <div className="field">
          <label htmlFor="mach-end">Open end</label>
          <select id="mach-end" value={m.free_end} disabled={disabled} onChange={(e) => onChange({ ...m, free_end: e.target.value as Machining['free_end'] })}>
            <option value="none">Neither (both ends held)</option>
            <option value="+">+{axis} end</option>
            <option value="-">−{axis} end</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="mach-tool">Smallest cutter Ø ({units})</label>
          <input id="mach-tool" type="number" min={0} step={units === 'in' ? 0.0625 : 1} value={tool} disabled={disabled}
            onChange={(e) => setTool(e.target.value)} onBlur={commitTool} onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />
        </div>
      </div>
      <p className="note">
        Profiles and flat-floored pockets cut from the {faces} faces (the rotary indexed every 90° about {axis}
        {m.free_end !== 'none' && ', plus the open end'}), walls parallel to the cutter, floors at {m.depths} equal depth
        {m.depths === 1 ? '' : 's'} from each face. The cutter size sets the smallest pocket and corner radius. Axes are the
        design-space (mount) frame; drilled bolt and pin holes are separate operations.
      </p>
    </>
  )
}

function MachinedToggle({ steps, stepIdx, onStep, mass }: { steps: TopoStep[]; stepIdx: number; onStep: (i: number) => void; mass: (kg: number) => string }) {
  const raw = steps[0]
  const mach = steps[1]
  // Both as 0/1 parts at the same mass, so the compliance ratio is the stiffness cost of the machining (averaged
  // over the load cases). Older results without solid values fall back to the optimiser densities.
  const solid = raw.solid_compliance?.length && mach.solid_compliance?.length
  const cr = solid ? raw.solid_compliance : raw.compliance
  const cm = solid ? mach.solid_compliance : mach.compliance
  const ratio = cm.reduce((a, c, i) => a + c / cr[i], 0) / cm.length
  const loss = (1 - 1 / ratio) * 100
  return (
    <>
      <div className="seg" role="group" aria-label="Result shown">
        <button aria-pressed={stepIdx === 0} onClick={() => onStep(0)}>Raw</button>
        <button aria-pressed={stepIdx === 1} onClick={() => onStep(1)}>Machinable</button>
      </div>
      <dl className="kv">
        <dt>Mill</dt><dd>{mach.label}</dd>
        <dt>Mass</dt><dd>{mass(solid ? mach.solid_mass : mach.mass)} (raw {mass(solid ? raw.solid_mass : raw.mass)})</dd>
        <dt>Stiffness</dt><dd className={loss > 15 ? 'wn' : ''}>{loss >= 0 ? `${loss.toFixed(1)}% below raw` : `${(-loss).toFixed(1)}% above raw`}</dd>
        <dt>Iterations</dt><dd>raw {raw.iterations}{raw.converged ? '' : '+'}, machinable {mach.iterations}{mach.converged ? '' : '+'}</dd>
      </dl>
      <p className="note">{solid ? 'Compared as solid parts at the same mass: the raw result cut by density, the machined part as cut.' : 'Compared on optimiser densities.'}</p>
    </>
  )
}

const LOG_CLASS: Record<InterpretJob['log'][number]['kind'], string> = { read: 'read', rule: 'rule', sketch: 'rule', ai: 'claude', build: 'build', skip: 'warn', check: 'read', question: 'question', done: 'finish', error: 'error' }
const LOG_ICON: Record<InterpretJob['log'][number]['kind'], string> = { read: '◇', rule: '§', sketch: '✎', ai: '✦', build: '▣', skip: '⚠', check: '⚖', question: '?', done: '✓', error: '✖' }

/** Turn the shown result into machined features: the button, the live log and the before/after card. */
function ParametricModel({ step, steps, stale, busy, job, hasDesign, showPart, onShowPart, showTrace, onShowTrace, onStart, onStop, mass, aiDraft }: {
  step: number; steps: TopoStep[]; stale: boolean; busy: boolean; job: InterpretJob | null; hasDesign: boolean
  showPart: boolean; onShowPart: (on: boolean) => void; showTrace: boolean; onShowTrace: (on: boolean) => void
  onStart: (step: number, mode?: 'rules' | 'ai', instruction?: string, maxBuilds?: number, stress?: boolean) => void; onStop: () => void
  mass: (kg: number) => string
  aiDraft: { summary: string; questions: string[] } | null
}) {
  const [confirm, setConfirm] = useState<null | 'rules' | 'ai'>(null)
  const asked = job?.status !== 'running' && aiDraft && aiDraft.questions.length > 0 ? aiDraft.questions : null
  const [ask, setAsk] = useState('')
  const [builds, setBuilds] = useState(5)
  const [withStress, setWithStress] = useState(false)
  const running = job?.status === 'running'
  const src = steps[step]
  const machined = src && src.label !== 'raw'
  const start = (mode: 'rules' | 'ai') => {
    setConfirm(null)
    onStart(step, mode, mode === 'ai' ? ask : '', builds, mode === 'ai' && withStress)
    if (mode === 'ai') setAsk('')
  }
  const tok = job?.tokens
  const c = job?.check
  return (
    <>
      <div className="sec">
        Parametric model{running && <span className="spin inline" aria-label="working" />}
      </div>
      <p className="note">
        Drafts sketches from the {machined ? 'machinable' : 'shown'} result, as a plate machined from ±Z: an outline of lugs and tangent lines
        with lightening holes, web pockets inside a rim with bosses at the pins, and spot faces so each lug fits its fork. Then pads and
        pockets them and compares the part with the result.
        {!machined && ' A raw result is rarely a clean plate; the machinable one reads best.'}
      </p>
      {asked && (
        <div className="ai-asked">
          <b>Claude asked about this design:</b>
          <ol>{asked.map((q, i) => <li key={i}>{q.replace(/^\d+\.\s*/, '')}</li>)}</ol>
          <span className="note">Answer below (by number is fine) and draft again: Claude starts from its last recipe and sees these questions.</span>
        </div>
      )}
      <div className="field">
        <label htmlFor="ai-ask">{asked ? 'Your answers or feedback for Claude' : 'For Claude: what you want, or feedback on its last draft'}</label>
        <textarea id="ai-ask" rows={3} value={ask} disabled={running} onChange={(e) => setAsk(e.target.value)}
          placeholder="e.g. Keep the curved outer edge; make the left side straight; aim for 0.35 lb" />
      </div>
      {confirm ? (
        <div className="sp-row">
          <button className="btn primary" onClick={() => start(confirm)}>Replace the current design</button>
          <button className="btn" onClick={() => setConfirm(null)}>Keep it</button>
        </div>
      ) : (
        <div className="sp-row">
          <button className="btn primary ai" disabled={!src || stale || busy || running} onClick={() => (hasDesign ? setConfirm('ai') : start('ai'))}
            title="Claude drafts the sketches from the result's measurements, builds and checks each draft, and asks you questions">
            ✦ Draft with AI
          </button>
          <select aria-label="Builds Claude may try" value={builds} disabled={running} onChange={(e) => setBuilds(+e.target.value)}>
            {[2, 3, 5, 8].map((n) => <option key={n} value={n}>{n} builds</option>)}
          </select>
          <button className="btn" disabled={!src || stale || busy || running} onClick={() => (hasDesign ? setConfirm('rules') : start('rules'))}
            title="The deterministic draft: hull of the lugs, filleted holes, web pockets, spot faces">
            Draft with rules
          </button>
          {running && job?.mode === 'ai' && <button className="btn danger" onClick={onStop}>Stop</button>}
        </div>
      )}
      <label className="sp-check" htmlFor="ai-stress" title="Each build also gets a coarse stress analysis (about 15 s more), so Claude sizes lugs, web and rim from the margins">
        <input id="ai-stress" type="checkbox" checked={withStress} disabled={running} onChange={(e) => setWithStress(e.target.checked)} /> Check stress on every AI build
      </label>
      {tok?.input ? <p className="note">Claude: {Math.round(tok.input / 1000)}k tokens in ({Math.round((tok.cache_read ?? 0) / 1000)}k cached), {Math.round((tok.output ?? 0) / 1000)}k out</p> : null}
      {stale && <p className="note wn">Run the optimisation again first; the set-up has changed.</p>}
      {job && (
        <>
          <ol className="ailog">
            {job.log.map((l, i) => (
              <li key={i} className={`ai-${LOG_CLASS[l.kind]}`}>
                <span className="ico" aria-hidden="true">{LOG_ICON[l.kind]}</span>
                <span className="tx">{l.text} <span className="aimeta">{l.t.toFixed(1)} s</span></span>
              </li>
            ))}
          </ol>
          {c && (
            <table className="holes topo">
              <thead><tr><th></th><th>Topology</th><th>Model</th><th>Δ</th></tr></thead>
              <tbody>
                <tr><td>Mass</td><td>{mass(c.ref_mass)}</td><td>{mass(c.mass)}</td><td>{fmtPct(c.mass / c.ref_mass - 1)}</td></tr>
                <tr><td>Stiffness</td><td>1.00</td><td>{c.stiffness_ratio.toFixed(2)}</td><td>{fmtPct(c.stiffness_ratio - 1)}</td></tr>
              </tbody>
            </table>
          )}
          {c && <p className="note">Both as solid voxels on the same mesh, same supports and load. The model’s exact mass is {mass(c.brep_mass)}.</p>}
          <label className="sp-check" htmlFor="topo-show-part">
            <input id="topo-show-part" type="checkbox" checked={showPart} onChange={(e) => onShowPart(e.target.checked)} /> Show the part (result as a ghost)
          </label>
          {job.overlay && (
            <>
              <label className="sp-check" htmlFor="topo-show-trace">
                <input id="topo-show-trace" type="checkbox" checked={showTrace} onChange={(e) => onShowTrace(e.target.checked)} /> Show sketches
              </label>
              {showTrace && (
                <p className="note legend">
                  {job.overlay.sketches.map((s, i) => (
                    <span key={s.id}><span className="dot" style={{ background: hex(SKETCH_COLORS[i % SKETCH_COLORS.length]) }} /> {s.id} {s.name} </span>
                  ))}
                </p>
              )}
            </>
          )}
        </>
      )}
    </>
  )
}

const hex = (c: number) => `#${c.toString(16).padStart(6, '0')}`
const fmtPct = (f: number) => `${f >= 0 ? '+' : ''}${(f * 100).toFixed(1)}%`
