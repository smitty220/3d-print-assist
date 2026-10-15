// The sketcher: a 2D view square to the sketch plane (as CATIA's Sketcher), over the topology thickness map.
// Drag points or curves (the solver runs live in the browser), click a dimension to change it, select items to constrain,
// split and join curves. Each finished edit goes to the backend, which solves with the same solver and rebuilds the part.
import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, type DesignState } from './api'
import { browserSolver } from './sketch/browser'
import { midpoint, pointOf, polyline, type XY } from './sketch/draw'
import { addConstraint, applicable, cornerRadius, deleteItems, join, setValue, splitLine, toggleConstruction } from './sketch/edit'
import type { Constraint, Entity, Sketch, SolveReport } from './sketch/model'
import { solveSketch, type Drag } from './sketch/solve'
import { len, type Units } from './units'

type Solver = Awaited<ReturnType<typeof browserSolver>>
type View = { cx: number; cy: number; k: number } // centre (mm) and px per mm
type Underlay = { url: string; x0: number; y0: number; w: number; h: number }

const GLYPH: Partial<Record<Constraint['kind'], string>> = {
  tangent: '⌒', perpendicular: '⊥', parallel: '∥', horizontal: 'H', vertical: 'V', concentric: '◎', equal: '=',
  coincident: '•', point_on: '⋯', fix: '⚓',
}
const NAME: Record<Constraint['kind'], string> = {
  coincident: 'Coincident', horizontal: 'Horizontal', vertical: 'Vertical', parallel: 'Parallel', perpendicular: 'Perpendicular',
  tangent: 'Tangent', concentric: 'Concentric', equal: 'Equal', point_on: 'Point on', distance: 'Distance', radius: 'Radius',
  diameter: 'Diameter', angle: 'Angle', fix: 'Fix',
}
const DIMS = new Set(['distance', 'radius', 'diameter', 'angle'])

interface Props {
  sketch: Sketch & { report?: SolveReport | null }
  others: Sketch[]
  units: Units
  propsHost: HTMLElement | null
  pocket: boolean // a pocket uses this sketch: offer to redraw it as an offset of a profile
  onDesign: (d: DesignState) => void
  onExit: () => void
}

export function SketchEditor({ sketch, others, units, propsHost, pocket, onDesign, onExit }: Props) {
  const [work, setWork] = useState<Sketch>(sketch)
  const [report, setReport] = useState<SolveReport | null>(sketch.report ?? null)
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [tool, setTool] = useState<'select' | 'split'>('select')
  const [undo, setUndo] = useState<Sketch[]>([])
  const [size, setSize] = useState({ w: 800, h: 600 })
  const [view, setView] = useState<View | null>(null)
  const [underlay, setUnderlay] = useState<Underlay | null>(null)
  const [edit, setEdit] = useState<{ cid: string; x: number; y: number; text: string } | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [solver, setSolver] = useState<Solver | null>(null)
  const [show, setShow] = useState({ constraints: true, dimensions: true })
  const [pendingDim, setPendingDim] = useState<string | null>(null) // open this dimension for typing once it lands
  const svgRef = useRef<SVGSVGElement>(null)
  const workRef = useRef(work)
  workRef.current = work
  const gesture = useRef<{ kind: 'pan' | 'drag'; x: number; y: number; view?: View; moved?: boolean; base?: Sketch; drag?: (p: XY) => Drag } | null>(null)
  const solving = useRef(false)

  useEffect(() => { browserSolver().then(setSolver).catch((e) => setMsg(`Solver failed to load: ${e}`)) }, [])
  useEffect(() => { setWork(sketch); setReport(sketch.report ?? null); setSel(new Set()); setUndo([]) }, [sketch.id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const el = svgRef.current?.parentElement
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Thickness of the topology result as the underlay (darker = thicker).
  useEffect(() => {
    api.topoUnderlay().then((u) => {
      const c = document.createElement('canvas')
      c.width = u.nx
      c.height = u.ny
      const ctx = c.getContext('2d')!
      const img = ctx.createImageData(u.nx, u.ny)
      for (let j = 0; j < u.ny; j++) for (let i = 0; i < u.nx; i++) {
        const t = u.thickness[j * u.nx + i] / u.nz
        const o = ((u.ny - 1 - j) * u.nx + i) * 4 // image rows run down, y runs up
        img.data.set(t > 0 ? [64, 120, 190, Math.round(40 + 150 * t)] : [0, 0, 0, 0], o)
      }
      ctx.putImageData(img, 0, 0)
      setUnderlay({ url: c.toDataURL(), x0: u.x0, y0: u.y0, w: u.nx * u.h, h: u.ny * u.h })
    }).catch(() => setUnderlay(null))
  }, [])

  // Fit once: the sketch and the underlay.
  useEffect(() => {
    if (view || !size.w) return
    const pts = work.entities.flatMap((e) => (e.kind === 'point' ? [[e.x, e.y] as XY] : polyline(work, e)))
    if (underlay) pts.push([underlay.x0, underlay.y0], [underlay.x0 + underlay.w, underlay.y0 + underlay.h])
    if (!pts.length) return
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
    setView({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, k: 0.85 * Math.min(size.w / (x1 - x0 || 1), size.h / (y1 - y0 || 1)) })
  }, [size, underlay, view, work])

  const v = view ?? { cx: 0, cy: 0, k: 1 }
  const sx = (x: number) => (x - v.cx) * v.k + size.w / 2
  const sy = (y: number) => size.h / 2 - (y - v.cy) * v.k
  const toMm = (px: number, py: number): XY => [(px - size.w / 2) / v.k + v.cx, (size.h / 2 - py) / v.k + v.cy]
  const path = (pl: XY[]) => pl.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('')
  const local = (e: React.PointerEvent | React.WheelEvent | PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top] as XY
  }

  const trouble = new Set([...(report?.conflicting ?? []), ...(report?.redundant ?? [])])
  const troubleRefs = new Set(work.constraints.filter((c) => trouble.has(c.id)).flatMap((c) => c.refs))
  // Annotations on curves that are only a few pixels on screen (fillets, zoomed out) wait until you zoom in, select
  // them or they are in trouble, as CATIA thins them out.
  const screenSize = (id: string) => {
    const e = work.entities.find((x) => x.id === id)
    if (!e || e.kind === 'point') return Infinity
    const pl = polyline(work, e)
    const xs = pl.map((p) => p[0]), ys = pl.map((p) => p[1])
    return Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * v.k
  }
  const visible = (c: Constraint) =>
    sel.has(c.id) || trouble.has(c.id) || c.refs.some((r) => sel.has(r)) || c.refs.every((r) => screenSize(r) >= 28)

  // ---------- edits ----------

  async function commit(next: Sketch, before: Sketch) {
    setBusy(true)
    setMsg(null)
    try {
      const r = await api.putSketch(next.id, next)
      setWork(r.sketch)
      setReport(r.sketch.report ?? null)
      setUndo((u) => [...u.slice(-49), before])
      onDesign(r)
      const bad = r.features.filter((f) => f.error)
      if (bad.length) setMsg(bad.map((f) => `${f.id}: ${f.error}`).join(' · '))
    } catch (e) {
      setWork(before)
      setMsg(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  const apply = (fn: (s: Sketch) => Sketch | string) => {
    const r = fn(work)
    if (typeof r === 'string') return setMsg(r)
    setSel(new Set())
    commit(r, work)
  }

  function undoLast() {
    const prev = undo[undo.length - 1]
    if (!prev) return
    setUndo((u) => u.slice(0, -1))
    commit(prev, work).then(() => setUndo((u) => u.slice(0, -1)))
  }

  // ---------- pointer ----------

  function startPan(e: React.PointerEvent) {
    gesture.current = { kind: 'pan', x: e.clientX, y: e.clientY, view: v }
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }

  function startDrag(e: React.PointerEvent, drag: (p: XY) => Drag) {
    if (e.button !== 0) return startPan(e)
    e.stopPropagation()
    gesture.current = { kind: 'drag', x: e.clientX, y: e.clientY, base: work, drag, moved: false }
    svgRef.current?.setPointerCapture(e.pointerId)
  }

  function onMove(e: React.PointerEvent) {
    const g = gesture.current
    if (!g) return
    if (g.kind === 'pan' && g.view) {
      setView({ ...g.view, cx: g.view.cx - (e.clientX - g.x) / g.view.k, cy: g.view.cy + (e.clientY - g.y) / g.view.k })
      return
    }
    if (g.kind === 'drag' && g.drag && solver && !solving.current) {
      if (Math.hypot(e.clientX - g.x, e.clientY - g.y) < 3 && !g.moved) return
      g.moved = true
      solving.current = true
      const target = g.drag(toMm(...local(e)))
      solveSketch(workRef.current, solver, target).then(({ sketch: s, report: r }) => {
        if (gesture.current === g) { setWork(s); setReport(r) }
      }).finally(() => { solving.current = false })
    }
  }

  function onUp() {
    const g = gesture.current
    gesture.current = null
    if (g?.kind === 'drag' && g.moved && g.base) {
      const before = g.base
      // Let the last live solve land, then hand the result to the backend.
      setTimeout(() => commit(workRef.current, before), 30)
    }
  }

  function onWheel(e: React.WheelEvent) {
    const [px, py] = local(e)
    const at = toMm(px, py)
    const k = v.k * Math.exp(-e.deltaY * 0.0015)
    setView({ k, cx: at[0] - (px - size.w / 2) / k, cy: at[1] + (py - size.h / 2) / k })
  }

  function pick(id: string, e: React.PointerEvent) {
    if (e.button !== 0) return
    setSel((cur) => {
      const n = new Set(e.shiftKey || e.metaKey || e.ctrlKey ? cur : [])
      if (n.has(id) && (e.shiftKey || e.metaKey || e.ctrlKey)) n.delete(id)
      else n.add(id)
      return n
    })
  }

  function onCurveDown(e: React.PointerEvent, ent: Entity) {
    if (e.button !== 0) return startPan(e)
    e.stopPropagation()
    const at = toMm(...local(e))
    if (tool === 'split') {
      if (ent.kind === 'line') apply((s) => splitLine(s, ent.id, at))
      else setMsg('Split works on lines; pick a straight segment')
      return
    }
    pick(ent.id, e)
    // Dragging a line moves both its ends; a circle's edge changes its radius.
    if (ent.kind === 'line') {
      const a = pointOf(work, ent.p1), b = pointOf(work, ent.p2)
      startDrag(e, (p) => ({ points: [{ id: ent.p1, x: a[0] + p[0] - at[0], y: a[1] + p[1] - at[1] }, { id: ent.p2, x: b[0] + p[0] - at[0], y: b[1] + p[1] - at[1] }] }))
    } else if (ent.kind === 'circle') {
      const c = pointOf(work, ent.c)
      startDrag(e, (p) => ({ radius: { id: ent.id, r: Math.hypot(p[0] - c[0], p[1] - c[1]) } }))
    }
  }

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return
      if ((e.key === 'Delete' || e.key === 'Backspace') && sel.size) { e.preventDefault(); apply((s) => deleteItems(s, sel)) }
      else if (e.key === 'z' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); undoLast() }
      else if (e.key === 'Escape') { if (tool !== 'select') setTool('select'); else if (sel.size) setSel(new Set()); else onExit() }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  })

  // ---------- constraint glyphs and dimensions ----------

  const entity = (id: string) => work.entities.find((e) => e.id === id)
  const ends = (e?: Entity) => (e?.kind === 'line' ? [e.p1, e.p2] : e?.kind === 'arc' ? [e.start, e.end] : [])
  const glyphs = useMemo(() => {
    const out: { c: Constraint; at: XY }[] = []
    for (const c of work.constraints) {
      if (!GLYPH[c.kind]) continue
      const [a, b] = c.refs.map((r) => work.entities.find((e) => e.id === r))
      try {
        if (c.kind === 'tangent' || c.kind === 'perpendicular') {
          const shared = ends(a).find((p) => ends(b).includes(p))
          out.push({ c, at: shared ? pointOf(work, shared) : midpoint(work, a!) })
        } else if (c.kind === 'parallel' || c.kind === 'equal') {
          out.push({ c, at: midpoint(work, a!) }, { c, at: midpoint(work, b!) })
        } else if (c.kind === 'horizontal' || c.kind === 'vertical') out.push({ c, at: midpoint(work, a!) })
        else if (c.kind === 'concentric') out.push({ c, at: pointOf(work, (a as { c: string }).c) })
        else out.push({ c, at: a?.kind === 'point' ? [a.x, a.y] : midpoint(work, a!) })
      } catch { /* a stale reference: shown in the list instead */ }
    }
    return out
  }, [work]) // eslint-disable-line react-hooks/exhaustive-deps

  function dimension(c: Constraint): { line?: [XY, XY]; at: XY; text: string } | null {
    try {
      const [a, b] = c.refs.map((r) => entity(r)!)
      const unit = units === 'in' ? '' : ' mm'
      const paren = (t: string) => (c.driving === false ? `(${t})` : t)
      if (c.kind === 'radius' || c.kind === 'diameter') {
        const centre = pointOf(work, (a as { c: string }).c)
        const m = a.kind === 'arc' ? midpoint(work, a) : ([centre[0] + (a as { r: number }).r * Math.SQRT1_2, centre[1] + (a as { r: number }).r * Math.SQRT1_2] as XY)
        const out: XY = [m[0] + (m[0] - centre[0]) * 0.25, m[1] + (m[1] - centre[1]) * 0.25]
        return { line: [centre, m], at: out, text: paren(`${c.kind === 'radius' ? 'R' : 'Ø'}${len(c.value ?? 0, units)}${unit}`) }
      }
      if (c.kind === 'distance') {
        const p = pointOf(work, a.id)
        let q: XY
        if (b.kind === 'line') {
          const q1 = pointOf(work, b.p1), q2 = pointOf(work, b.p2)
          const vx = q2[0] - q1[0], vy = q2[1] - q1[1]
          const t = ((p[0] - q1[0]) * vx + (p[1] - q1[1]) * vy) / (vx * vx + vy * vy)
          q = [q1[0] + t * vx, q1[1] + t * vy]
        } else q = pointOf(work, b.id)
        return { line: [p, q], at: [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], text: paren(`${len(c.value ?? 0, units)}${unit}`) }
      }
      if (c.kind === 'angle') return { at: midpoint(work, a), text: paren(`${(((c.value ?? 0) * 180) / Math.PI).toFixed(1)}°`) }
    } catch { /* stale */ }
    return null
  }

  function openDim(c: Constraint, px: number, py: number) {
    const shown = c.kind === 'angle' ? (((c.value ?? 0) * 180) / Math.PI).toFixed(2) : len(c.value ?? 0, units, units === 'in' ? 4 : 3)
    setEdit({ cid: c.id, x: px, y: py, text: shown })
  }

  function closeDim(save: boolean) {
    if (save && edit) {
      const c = work.constraints.find((k) => k.id === edit.cid)!
      const n = Number(edit.text)
      if (Number.isFinite(n) && (c.kind === 'angle' || n > 0)) {
        const value = c.kind === 'angle' ? (n * Math.PI) / 180 : units === 'in' ? n * 25.4 : n
        apply((s) => setValue(s, c.id, value))
      } else setMsg('Enter a positive number')
    }
    setEdit(null)
  }

  // A new corner radius: open its dimension so the value can be typed straight away.
  useEffect(() => {
    if (!pendingDim || busy) return
    const c = work.constraints.find((k) => k.id === pendingDim)
    setPendingDim(null)
    const dm = c && dimension(c)
    if (c && dm) openDim(c, sx(dm.at[0]), sy(dm.at[1]))
  }, [work, busy, pendingDim]) // eslint-disable-line react-hooks/exhaustive-deps

  function corner() {
    const lines = selIds.filter((id) => entity(id)?.kind === 'line')
    if (lines.length !== 2) return setMsg('Select two lines')
    const r = cornerRadius(work, lines[0], lines[1], units === 'in' ? 0.125 * 25.4 : 3)
    if (typeof r === 'string') return setMsg(r)
    setSel(new Set())
    setPendingDim(r.dim)
    commit(r.sketch, work)
  }

  // ---------- render ----------

  const selIds = [...sel]
  const options = applicable(work, selIds.filter((id) => entity(id)))
  const stacked = new Map<string, number>()
  const curveIds = work.entities.filter((e) => e.kind !== 'point').map((e) => e.id)

  const canvas = (
    <div className="sketcher" onContextMenu={(e) => e.preventDefault()}>
      <svg ref={svgRef} width={size.w} height={size.h} onPointerDown={(e) => { if (e.button === 0 && tool === 'select') setSel(new Set()); if (e.button !== 0) startPan(e) }}
        onPointerMove={onMove} onPointerUp={onUp} onWheel={onWheel} className={tool === 'split' ? 'split' : ''}>
        {underlay && (
          <image href={underlay.url} x={sx(underlay.x0)} y={sy(underlay.y0 + underlay.h)} width={underlay.w * v.k} height={underlay.h * v.k}
            preserveAspectRatio="none" style={{ imageRendering: 'pixelated' }} />
        )}
        {others.map((o) => o.entities.filter((e) => e.kind !== 'point' && !('construction' in e && e.construction)).map((e) => (
          <path key={`${o.id}-${e.id}`} d={path(polyline(o, e))} className="sk-other" />
        )))}
        {work.entities.filter((e) => e.kind !== 'point').map((e) => {
          const cls = `sk-curve${'construction' in e && e.construction ? ' cons' : ''}${sel.has(e.id) ? ' sel' : ''}${troubleRefs.has(e.id) ? ' bad' : ''}`
          const d = path(polyline(work, e))
          return (
            <g key={e.id}>
              <path d={d} className={cls} />
              <path d={d} className="sk-hit" onPointerDown={(ev) => onCurveDown(ev, e)} />
            </g>
          )
        })}
        {show.dimensions && work.constraints.filter((c) => DIMS.has(c.kind) && visible(c)).map((c) => {
          const dm = dimension(c)
          if (!dm) return null
          const bad = trouble.has(c.id)
          return (
            <g key={c.id} className={`sk-dim${sel.has(c.id) ? ' sel' : ''}${bad ? ' bad' : ''}`}>
              {dm.line && <line x1={sx(dm.line[0][0])} y1={sy(dm.line[0][1])} x2={sx(dm.line[1][0])} y2={sy(dm.line[1][1])} />}
              <text x={sx(dm.at[0])} y={sy(dm.at[1])} onPointerDown={(e) => { e.stopPropagation(); pick(c.id, e) }}
                onDoubleClick={() => openDim(c, sx(dm.at[0]), sy(dm.at[1]))} onClick={(e) => { if (!e.shiftKey) openDim(c, sx(dm.at[0]), sy(dm.at[1])) }}>
                {dm.text}
              </text>
            </g>
          )
        })}
        {show.constraints && glyphs.filter(({ c }) => visible(c)).map(({ c, at }, i) => {
          const key = `${Math.round(sx(at[0]) / 6)},${Math.round(sy(at[1]) / 6)}`
          const n = stacked.get(key) ?? 0
          stacked.set(key, n + 1)
          return (
            <g key={`${c.id}-${i}`} className={`sk-glyph${sel.has(c.id) ? ' sel' : ''}${trouble.has(c.id) ? ' bad' : ''}`}
              transform={`translate(${sx(at[0]) + 9 + n * 15},${sy(at[1]) - 9})`} onPointerDown={(e) => { e.stopPropagation(); pick(c.id, e) }}>
              <title>{`${NAME[c.kind]} (${c.id})${trouble.has(c.id) ? ' — conflicting or redundant' : ''}`}</title>
              <rect x={-7} y={-7} width={14} height={14} rx={3} />
              <text>{GLYPH[c.kind]}</text>
            </g>
          )
        })}
        {work.entities.filter((e) => e.kind === 'point').map((p) => {
          if (p.kind !== 'point') return null
          const used = work.entities.some((e) => e.kind !== 'point' && ([...ends(e), 'c' in e ? e.c : '', ...(e.kind === 'spline' ? e.points : [])] as string[]).includes(p.id))
          if (!used && !p.ref) return null
          const x = sx(p.x), y = sy(p.y)
          return p.ref ? (
            <g key={p.id} className="sk-pin" transform={`translate(${x},${y})`}>
              <title>{`${p.id}: held at ${p.ref}`}</title>
              <circle r={5} /><path d="M-7,0H7M0,-7V7" />
            </g>
          ) : (
            <rect key={p.id} x={x - 3.5} y={y - 3.5} width={7} height={7} className={`sk-pt${sel.has(p.id) ? ' sel' : ''}${p.fixed ? ' fixed' : ''}`}
              onPointerDown={(e) => { pick(p.id, e); startDrag(e, (q) => ({ points: [{ id: p.id, x: q[0], y: q[1] }] })) }} />
          )
        })}
      </svg>
      {edit && (
        <input className="sk-input" autoFocus style={{ left: edit.x - 40, top: edit.y - 14 }} value={edit.text}
          onChange={(e) => setEdit({ ...edit, text: e.target.value })} onBlur={() => closeDim(true)}
          onKeyDown={(e) => { if (e.key === 'Enter') closeDim(true); if (e.key === 'Escape') closeDim(false) }} />
      )}
      <div className="sk-tools">
        <button className="btn primary" onClick={onExit} title="Back to 3D (Esc)">Exit sketch</button>
        <button className="btn" onClick={undoLast} disabled={!undo.length || busy} title="Undo (Ctrl/⌘ Z)">Undo</button>
        <button className="btn" aria-pressed={tool === 'split'} onClick={() => setTool(tool === 'split' ? 'select' : 'split')} title="Click a line to break it in two">Split line</button>
        <button className="btn" disabled={sel.size < 2} onClick={() => apply((s) => join(s, selIds.filter((id) => curveIds.includes(id)), 'line'))}>Join → line</button>
        <button className="btn" disabled={sel.size < 2} onClick={() => apply((s) => join(s, selIds.filter((id) => curveIds.includes(id)), 'spline'))}>Join → spline</button>
        <button className="btn" disabled={selIds.filter((id) => entity(id)?.kind === 'line').length !== 2} onClick={corner}
          title="Round the corner between two selected lines with an arc tangent to both (lines are trimmed or extended to it)">Corner radius</button>
        <button className="btn" disabled={!selIds.some((id) => curveIds.includes(id))} onClick={() => apply((s) => toggleConstruction(s, sel))}>Construction</button>
        <button className="btn danger" disabled={!sel.size} onClick={() => apply((s) => deleteItems(s, sel))}>Delete</button>
        {selIds.some((id) => entity(id)?.kind === 'spline') && (
          <button className="btn" title="Make the selected splines leave and arrive along the curves they join (smooth), or free"
            onClick={() => apply((s) => ({ ...s, entities: s.entities.map((e) => {
              if (e.kind !== 'spline' || !sel.has(e.id)) return e
              const on = !(e.tangent_start && e.tangent_end)
              return { ...e, tangent_start: on, tangent_end: on }
            }) }))}>Tangent ends</button>
        )}
        <label className="sk-toggle"><input type="checkbox" checked={show.constraints} onChange={(e) => setShow({ ...show, constraints: e.target.checked })} /> Constraints</label>
        <label className="sk-toggle"><input type="checkbox" checked={show.dimensions} onChange={(e) => setShow({ ...show, dimensions: e.target.checked })} /> Dimensions</label>
        {options.map((o) => (
          <button key={o.kind + o.label} className="btn ai" onClick={() => apply((s) => addConstraint(s, o.kind, selIds))}>{o.label}</button>
        ))}
      </div>
      <div className={`sk-status${report && (report.conflicting.length || report.redundant.length || report.status === 'failed') ? ' bad' : ''}`}>
        {busy ? 'Solving and rebuilding…' : statusText(report)}{msg && <span className="no"> · {msg}</span>}
      </div>
    </div>
  )

  return (
    <>
      {canvas}
      {propsHost && createPortal(
        <SketchProps sketch={work} report={report} units={units} sel={sel} onSelect={(id) => setSel(new Set([id]))}
          onDelete={(id) => apply((s) => deleteItems(s, new Set([id])))} onRename={(name) => apply((s) => ({ ...s, name }))}
          onEditDim={(c) => { const dm = dimension(c); if (dm) openDim(c, sx(dm.at[0]), sy(dm.at[1])) }}
          sources={pocket ? others : []} onOffset={offsetFrom} busy={busy} />,
        propsHost)}
    </>
  )

  async function offsetFrom(source: string, rim: number, gap: number) {
    setBusy(true)
    setMsg(null)
    const before = work
    try {
      const r = await api.offsetSketch(work.id, source, rim, gap)
      setWork(r.sketch)
      setReport(r.sketch.report ?? null)
      setUndo((u) => [...u.slice(-49), before])
      setSel(new Set())
      onDesign(r)
    } catch (e) {
      setMsg(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  function statusText(r: SolveReport | null) {
    if (!r) return 'Not solved yet'
    if (r.status === 'failed') return 'The solver could not satisfy every constraint'
    const parts = [r.dof === 0 ? 'Fully constrained' : `${r.dof} degree${r.dof === 1 ? '' : 's'} of freedom (under-constrained: drag to explore)`]
    if (r.conflicting.length) parts.push(`conflicting: ${r.conflicting.join(', ')}`)
    if (r.redundant.length) parts.push(`redundant: ${r.redundant.join(', ')}`)
    return parts.join(' · ')
  }
}

function SketchProps({ sketch, report, units, sel, onSelect, onDelete, onRename, onEditDim, sources, onOffset, busy }: {
  sketch: Sketch; report: SolveReport | null; units: Units; sel: Set<string>
  onSelect: (id: string) => void; onDelete: (id: string) => void; onRename: (name: string) => void; onEditDim: (c: Constraint) => void
  sources: Sketch[]; onOffset: (source: string, rim: number, gap: number) => void; busy: boolean
}) {
  const toMm = (v: number) => (units === 'in' ? v * 25.4 : v)
  const [src, setSrc] = useState(sources.find((s) => /profile/i.test(s.name))?.id ?? sources[0]?.id ?? '')
  const [rim, setRim] = useState(units === 'in' ? '0.150' : '3.8')
  const [gap, setGap] = useState('0')
  const [name, setName] = useState(sketch.name)
  useEffect(() => setName(sketch.name), [sketch.name])
  const trouble = new Set([...(report?.conflicting ?? []), ...(report?.redundant ?? [])])
  const curves = sketch.entities.filter((e) => e.kind !== 'point')
  const value = (c: Constraint) => {
    if (c.value == null) return ''
    if (c.kind === 'angle') return `${((c.value * 180) / Math.PI).toFixed(1)}°`
    return `${len(c.value, units)} ${units}`
  }
  return (
    <>
      <h3>Sketch {sketch.id}</h3>
      <div className="field">
        <label htmlFor="sk-name">Name</label>
        <input id="sk-name" value={name} onChange={(e) => setName(e.target.value)} onBlur={() => name !== sketch.name && onRename(name)} />
      </div>
      <p className="note">
        {curves.length} curves, {sketch.constraints.length} constraints
        {report ? `, ${report.dof === 0 ? 'fully constrained' : `${report.dof} degrees of freedom`}` : ''}.
      </p>
      <p className="note">
        Drag points or curves; the solver keeps every constraint while you drag. Click a dimension to type a new value.
        Select items (Shift adds) to see the constraints that fit them; Delete removes the selection. Pins (⊕) are held by the
        interfaces. Middle or right drag pans, the wheel zooms.
      </p>
      {sources.length > 0 && (
        <>
          <div className="sec">Offset from profile</div>
          <p className="note">Redraw this sketch a standard offset inside another: its outer edge moved in by the rim, clear of a
            boss at each lug (the lug's radius plus the gap), outside corners at the cutter radius.</p>
          <div className="row2">
            <div className="field">
              <label htmlFor="sk-src">From</label>
              <select id="sk-src" value={src} onChange={(e) => setSrc(e.target.value)}>
                {sources.map((s) => <option key={s.id} value={s.id}>{s.id} {s.name}</option>)}
              </select>
            </div>
            <div className="field">
              <label htmlFor="sk-rim">Rim ({units})</label>
              <input id="sk-rim" value={rim} onChange={(e) => setRim(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="sk-gap">Boss gap ({units})</label>
              <input id="sk-gap" value={gap} onChange={(e) => setGap(e.target.value)} />
            </div>
          </div>
          <button className="btn" disabled={busy || !src || !(Number(rim) > 0)} onClick={() => onOffset(src, toMm(Number(rim)), toMm(Number(gap) || 0))}>
            Redraw as offset of {src}
          </button>
        </>
      )}
      <div className="sec">Constraints</div>
      <table className="holes sk-list">
        <tbody>
          {sketch.constraints.map((c) => (
            <tr key={c.id} className={`${sel.has(c.id) ? 'sel' : ''}${trouble.has(c.id) ? ' bad' : ''}`} onClick={() => onSelect(c.id)}>
              <td>{GLYPH[c.kind] ?? (c.kind === 'radius' ? 'R' : c.kind === 'diameter' ? 'Ø' : c.kind === 'angle' ? '∠' : '↔')}</td>
              <td>{NAME[c.kind]}</td>
              <td className="refs">{c.refs.join(', ')}</td>
              <td>{DIMS.has(c.kind) ? <button className="link" onClick={(e) => { e.stopPropagation(); onEditDim(c) }}>{value(c)}</button> : ''}</td>
              <td><button className="x" title="Delete constraint" onClick={(e) => { e.stopPropagation(); onDelete(c.id) }}>×</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      {trouble.size > 0 && <p className="note wn">Red rows are conflicting or redundant: delete one to free the sketch.</p>}
    </>
  )
}

