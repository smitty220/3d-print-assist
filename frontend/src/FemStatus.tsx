// A floating window that follows a running stress job: the stages done and in hand, and for each solve the conjugate-
// gradient residual against its convergence tolerance, iteration by iteration.
import { useEffect, useRef, useState } from 'react'
import type { FemJob } from './api'

const STAGES: { key: string; label: string }[] = [
  { key: 'mesh', label: 'Mesh' },
  { key: 'loads', label: 'Loads (statics, bearing)' },
  { key: 'assemble', label: 'Assemble stiffness' },
  { key: 'precondition', label: 'AMG preconditioner' },
  { key: 'iterate', label: 'Conjugate gradient' },
  { key: 'contact', label: 'Contact passes' },
  { key: 'stress', label: 'Stress recovery' },
]
const TITLE: Record<FemJob['mode'], string> = { analyse: 'Solving', mesh: 'Meshing', plan: 'Claude planning the mesh', size: 'Sizing the part' }

export function FemStatus({ job, onClose }: { job: FemJob; onClose: () => void }) {
  const p = job.progress ?? {}
  const running = job.status === 'running'
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    if (!running) return
    const t = window.setInterval(() => setNow(Date.now() / 1000), 250)
    return () => window.clearInterval(t)
  }, [running])
  const elapsed = (job.finished ?? now) - job.started
  const res = p.residuals ?? []
  const rtol = p.rtol ?? 1e-8
  const cur = p.stage ?? 'mesh'
  // Stages seen in the current run (a convergence check or a sizing trial starts the list again).
  const runStages = (p.stages ?? []).filter((s) => s.run === (p.run ?? ''))
  const seen = new Set(runStages.map((s) => s.name))
  const order = STAGES.map((s) => s.key)
  const curIdx = order.indexOf(cur)
  const last = job.log[job.log.length - 1]

  // Drag by the title bar.
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null)
  const drag = useRef<{ dx: number; dy: number } | null>(null)
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const move = (e: MouseEvent) => { if (drag.current) setPos({ x: e.clientX - drag.current.dx, y: e.clientY - drag.current.dy }) }
    const up = () => { drag.current = null }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
  }, [])

  return (
    <div ref={box} className="fem-status" role="dialog" aria-label="Solve status" style={pos ? { left: pos.x, top: pos.y, right: 'auto' } : undefined}>
      <div className="fs-head" onMouseDown={(e) => {
        const r = box.current!.getBoundingClientRect()
        const parent = box.current!.offsetParent!.getBoundingClientRect()
        drag.current = { dx: e.clientX - (r.left - parent.left), dy: e.clientY - (r.top - parent.top) }
      }}>
        <b>{TITLE[job.mode]}{p.run && job.mode === 'analyse' ? ` · ${p.run}` : ''}</b>
        <span className={`fs-state ${job.status}`}>{running ? 'running' : job.status === 'done' ? 'finished' : 'failed'} · {elapsed.toFixed(0)} s</span>
        <button className="x" aria-label="Close" onClick={onClose}>×</button>
      </div>
      {job.mode !== 'plan' && job.mode !== 'mesh' && (
        <ol className="fs-stages">
          {STAGES.map((s, i) => {
            const done = !running ? job.status === 'done' || i < curIdx : seen.has(s.key) && i < curIdx
            const now_ = running && s.key === cur
            return (
              <li key={s.key} className={done ? 'done' : now_ ? 'now' : ''}>
                <span className="dot" aria-hidden="true">{done ? '✓' : now_ ? '●' : '○'}</span>
                {s.label}
                {s.key === 'precondition' && p.levels ? <span className="aimeta"> {p.levels} levels</span> : null}
                {s.key === 'iterate' && res.length ? <span className="aimeta"> {res.length} iterations</span> : null}
                {s.key === 'contact' && p.contact?.length ? <span className="aimeta"> pass {p.contact[p.contact.length - 1].pass}: {p.contact[p.contact.length - 1].active.toLocaleString()} points in contact, {p.contact[p.contact.length - 1].changed} changed</span> : null}
              </li>
            )
          })}
        </ol>
      )}
      {(job.mode === 'analyse' || job.mode === 'size') && (
        <>
          <div className="fs-meta">
            {p.case && <span>Load case {p.case}</span>}
            {p.dof ? <span>{p.dof.toLocaleString()} DOF</span> : null}
            <span>Converged when |f − Ku| / |f| &lt; {rtol.toExponential(0)}</span>
          </div>
          <ResidualPlot res={res} rtol={rtol} />
          <div className="fs-meta">
            {res.length > 0 ? (
              <span className={res[res.length - 1] < rtol ? 'ok' : ''}>
                Iteration {res.length}: residual {res[res.length - 1].toExponential(2)} {res[res.length - 1] < rtol ? '< tolerance ✓ converged' : `(${Math.max(0, Math.log10(res[res.length - 1] / rtol)).toFixed(1)} decades to go)`}
              </span>
            ) : <span className="note">{cur === 'mesh' ? 'Meshing…' : cur === 'iterate' ? 'Starting iterations…' : 'Waiting for the solver…'}</span>}
          </div>
          {(p.solves ?? []).length > 0 && (
            <table className="holes topo fs-solves">
              <thead><tr><th>Solve</th><th>Iterations</th><th>Final residual</th></tr></thead>
              <tbody>
                {(p.solves ?? []).slice(-6).map((s, i) => (
                  <tr key={i}><td>{s.run} {s.case}</td><td>{s.iterations}</td><td className={s.residual < s.rtol ? 'ok' : 'no'}>{s.residual.toExponential(2)}</td></tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
      {p.peaks && p.peaks.spots.length > 0 && (
        <table className="holes topo fs-solves">
          <thead><tr><th>Hot spot</th><th>Peak by mesh (MPa)</th><th /></tr></thead>
          <tbody>
            {p.peaks.spots.map((s) => (
              <tr key={s.id}><td>{s.region || s.id}</td><td>{s.history.map((h) => (h.vm * s.factor).toFixed(0)).join(' → ')}</td>
                <td className={s.status === 'converged' ? 'ok' : s.status === 'singular' ? 'no' : ''}>{s.status === 'converging' ? `round ${p.peaks!.round}` : s.status}</td></tr>
            ))}
          </tbody>
        </table>
      )}
      {last && <p className={`note fs-last ${last.kind === 'error' ? 'no' : ''}`}>{last.text}</p>}
    </div>
  )
}

/** log10(residual) against iteration, with the tolerance as a dashed line. */
function ResidualPlot({ res, rtol }: { res: number[]; rtol: number }) {
  const W = 300, H = 120, padL = 34, padB = 16, padT = 6, padR = 6
  const lo = Math.min(Math.log10(rtol) - 1, ...res.map((r) => Math.log10(Math.max(r, 1e-30))))
  const hi = Math.max(0, ...res.map((r) => Math.ceil(Math.log10(Math.max(r, 1e-30)))))
  const n = Math.max(10, res.length)
  const x = (i: number) => padL + ((W - padL - padR) * i) / n
  const y = (r: number) => padT + ((H - padT - padB) * (hi - Math.log10(Math.max(r, 1e-30)))) / (hi - lo)
  const pts = [1, ...res].map((r, i) => `${x(i).toFixed(1)},${y(r).toFixed(1)}`).join(' ')
  const decades = []
  for (let d = hi; d >= Math.ceil(lo); d -= 2) decades.push(d)
  return (
    <svg className="fs-plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Residual history: ${res.length} iterations`}>
      {decades.map((d) => (
        <g key={d}>
          <line x1={padL} x2={W - padR} y1={y(10 ** d)} y2={y(10 ** d)} className="grid" />
          <text x={padL - 4} y={y(10 ** d) + 3} textAnchor="end">1e{d}</text>
        </g>
      ))}
      <line x1={padL} x2={W - padR} y1={y(rtol)} y2={y(rtol)} className="tol" />
      <text x={W - padR} y={y(rtol) - 3} textAnchor="end" className="tol-l">tolerance</text>
      <polyline points={pts} className="res" />
      <text x={W - padR} y={H - 3} textAnchor="end">iteration {res.length}</text>
    </svg>
  )
}
