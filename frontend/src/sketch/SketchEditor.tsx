// The sketcher: a 2D view square to the sketch plane. Draw with the sketch toolbar (line, rectangle, circle, arc,
// polygon, slot, point), constrain the selection, and dimension: a dimension takes a number, a parameter or a formula.
// Drag points or curves (the solver runs live in the browser); each finished edit goes to the backend, which solves it
// with the same solver and saves it. The sketch toolbar and the sketch's own tree mount in the main layout.
import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, type ParamNode, type Tree } from '../api'
import { ExprField } from '../edit/ExprField'
import { Icon } from '../ui/icons'
import { browserSolver } from './browser'
import { midpoint, pointOf, polyline, type XY } from './draw'
import { addConstraint, applicable, cornerRadius, deleteItems, join, splitLine, toggleConstruction } from './edit'
import type { Constraint, Entity, Sketch, SolveReport } from './model'
import { solveSketch, type Drag } from './solve'
import { addArc, addCenterRectangle, addCircle, addDimension, addLine, addPoint, addPolygon, addRectangle, addSlot, nearestPoint, type PointRef } from './tools'

type Solver = Awaited<ReturnType<typeof browserSolver>>
type View = { cx: number; cy: number; k: number } // centre (mm) and px per mm
type ToolId = 'select' | 'line' | 'rect' | 'crect' | 'circle' | 'arc' | 'polygon' | 'slot' | 'point' | 'split' | 'dimension'

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
const DRAW: { id: ToolId; label: string; icon: string; clicks: number; hint: string }[] = [
  { id: 'line', label: 'Line', icon: 'line', clicks: 0, hint: 'Click points; each click adds a line. Double-click or Esc to stop.' },
  { id: 'rect', label: 'Rectangle', icon: 'rect', clicks: 2, hint: 'Click two opposite corners.' },
  { id: 'crect', label: 'Center rect', icon: 'centerRect', clicks: 2, hint: 'Click the centre, then a corner.' },
  { id: 'circle', label: 'Circle', icon: 'circle', clicks: 2, hint: 'Click the centre, then a point on the circle.' },
  { id: 'arc', label: 'Arc', icon: 'arc', clicks: 3, hint: 'Click the centre, the start, then the end (counter-clockwise).' },
  { id: 'polygon', label: 'Polygon', icon: 'polygon', clicks: 2, hint: 'Click the centre, then a corner.' },
  { id: 'slot', label: 'Slot', icon: 'slot', clicks: 3, hint: 'Click both centres, then a point on the edge for the width.' },
  { id: 'point', label: 'Point', icon: 'point', clicks: 1, hint: 'Click to place a point.' },
]

interface Props {
  sid: string
  tree: Tree
  params: ParamNode[]
  toolbarHost: HTMLElement | null
  treeHost: HTMLElement | null
  onChanged: (t: Tree) => void
  onExit: () => void
}

export function SketchEditor({ sid, tree, params, toolbarHost, treeHost, onChanged, onExit }: Props) {
  const [work, setWork] = useState<Sketch | null>(null)
  const [report, setReport] = useState<SolveReport | null>(null)
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [tool, setTool] = useState<ToolId>('select')
  const [clicks, setClicks] = useState<{ at: XY; id: string | null }[]>([])
  const [hover, setHover] = useState<XY | null>(null)
  const [sides, setSides] = useState(6)
  const [undo, setUndo] = useState<Sketch[]>([])
  const [size, setSize] = useState({ w: 800, h: 600 })
  const [view, setView] = useState<View | null>(null)
  const [edit, setEdit] = useState<{ cid: string; x: number; y: number } | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [solver, setSolver] = useState<Solver | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const workRef = useRef(work)
  workRef.current = work
  const gesture = useRef<{ kind: 'pan' | 'drag'; x: number; y: number; view?: View; moved?: boolean; base?: Sketch; drag?: (p: XY) => Drag } | null>(null)
  const solving = useRef(false)

  useEffect(() => { browserSolver().then(setSolver).catch((e) => setMsg(`Solver failed to load: ${e}`)) }, [])
  useEffect(() => {
    api.sketch(sid).then((s) => { setWork(s); setReport(s.report ?? null); setSel(new Set()); setUndo([]); setView(null) })
      .catch((e) => setMsg(String(e)))
  }, [sid])
  useEffect(() => {
    const el = svgRef.current?.parentElement
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [work === null]) // eslint-disable-line react-hooks/exhaustive-deps

  // Fit once to the geometry (an empty sketch shows ±60 mm around the origin).
  useEffect(() => {
    if (view || !size.w || !work) return
    const pts = work.entities.flatMap((e) => (e.kind === 'point' ? [[e.x, e.y] as XY] : polyline(work, e)))
    if (!pts.length) pts.push([-60, -60], [60, 60])
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
    setView({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, k: 0.62 * Math.min(size.w / (x1 - x0 || 1), size.h / (y1 - y0 || 1)) })
  }, [size, view, work])

  const v = view ?? { cx: 0, cy: 0, k: 1 }
  const sx = (x: number) => (x - v.cx) * v.k + size.w / 2
  const sy = (y: number) => size.h / 2 - (y - v.cy) * v.k
  const toMm = (px: number, py: number): XY => [(px - size.w / 2) / v.k + v.cx, (size.h / 2 - py) / v.k + v.cy]
  const path = (pl: XY[]) => pl.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('')
  const local = (e: React.PointerEvent | React.WheelEvent | React.MouseEvent) => {
    const r = svgRef.current!.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top] as XY
  }

  // ---------- edits ----------

  async function commit(next: Sketch, before: Sketch, then?: (s: Sketch) => void) {
    setBusy(true)
    setMsg(null)
    try {
      const r = await api.putSketch(next.id, { name: next.name, entities: next.entities, constraints: next.constraints })
      setWork(r.sketch)
      setReport(r.sketch.report ?? null)
      setUndo((u) => [...u.slice(-49), before])
      onChanged(r.tree)
      then?.(r.sketch)
    } catch (e) {
      setWork(before)
      setMsg(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  const apply = (fn: (s: Sketch) => Sketch | string) => {
    if (!work) return
    const r = fn(work)
    if (typeof r === 'string') return setMsg(r)
    setSel(new Set())
    commit(r, work)
  }

  function undoLast() {
    const prev = undo[undo.length - 1]
    if (!prev || !work) return
    commit(prev, work).then(() => setUndo((u) => u.slice(0, -2)))
  }

  // ---------- drawing ----------

  const snapTol = () => 8 / v.k
  function snapped(at: XY): { at: XY; id: string | null } {
    const id = work ? nearestPoint(work, at, snapTol()) : null
    const p = id && work ? pointOf(work, id) : at
    return { at: p, id }
  }
  const ref = (c: { at: XY; id: string | null }): PointRef => (c.id ? { id: c.id } : { at: c.at })

  function drawClick(at0: XY, dbl = false) {
    if (!work) return
    const c = snapped(at0)
    const pts = [...clicks, c]
    const spec = DRAW.find((d) => d.id === tool)!
    try {
      if (tool === 'line') {
        const prev = pts[pts.length - 2]
        if (dbl || (prev && Math.hypot(prev.at[0] - c.at[0], prev.at[1] - c.at[1]) < snapTol())) { setClicks([]); return }
        if (pts.length === 1) { setClicks(pts); return }
        const r = addLine(work, ref(pts[pts.length - 2]), ref(c))
        commit(r.sketch, work, (s) => {
          const end = s.entities.find((e) => e.id === r.end)
          setClicks(end?.kind === 'point' ? [{ at: [end.x, end.y], id: r.end }] : [])
        })
        return
      }
      if (pts.length < spec.clicks) { setClicks(pts); return }
      setClicks([])
      const [a, b, d] = pts
      let next: Sketch
      if (tool === 'rect') next = addRectangle(work, a.at, b.at).sketch
      else if (tool === 'crect') next = addCenterRectangle(work, a.at, b.at).sketch
      else if (tool === 'circle') next = addCircle(work, ref(a), Math.hypot(b.at[0] - a.at[0], b.at[1] - a.at[1])).sketch
      else if (tool === 'arc') next = addArc(work, ref(a), ref(b), d.at).sketch
      else if (tool === 'polygon') next = addPolygon(work, a.at, b.at, sides).sketch
      else if (tool === 'slot') {
        const L = Math.hypot(b.at[0] - a.at[0], b.at[1] - a.at[1]) || 1
        const w = 2 * Math.abs(((d.at[0] - a.at[0]) * (b.at[1] - a.at[1]) - (d.at[1] - a.at[1]) * (b.at[0] - a.at[0])) / L)
        next = addSlot(work, a.at, b.at, w).sketch
      } else next = addPoint(work, a.at).sketch
      commit(next, work)
    } catch (e) {
      setClicks([])
      setMsg(String(e instanceof Error ? e.message : e))
    }
  }

  function preview(): XY[][] {
    if (!hover || !clicks.length) return []
    const h = snapped(hover).at
    const [a, b] = clicks.map((c) => c.at)
    if (tool === 'line') return [[a, h]]
    if (tool === 'rect') return [[a, [h[0], a[1]], h, [a[0], h[1]], a]]
    if (tool === 'crect') { const o: XY = [2 * a[0] - h[0], 2 * a[1] - h[1]]; return [[o, [h[0], o[1]], h, [o[0], h[1]], o]] }
    const ring = (c: XY, r: number, n = 48, t0 = 0, t1 = 2 * Math.PI) => Array.from({ length: n + 1 }, (_, i) => {
      const t = t0 + ((t1 - t0) * i) / n
      return [c[0] + r * Math.cos(t), c[1] + r * Math.sin(t)] as XY
    })
    if (tool === 'circle') return [ring(a, Math.hypot(h[0] - a[0], h[1] - a[1]))]
    if (tool === 'polygon') {
      const r = Math.hypot(h[0] - a[0], h[1] - a[1])
      const t0 = -Math.PI / 2 - Math.PI / sides
      return [ring(a, r, sides, t0, t0 + 2 * Math.PI), ring(a, r)]
    }
    if (tool === 'arc' && !b) return [[a, h]]
    if (tool === 'arc') {
      const r = Math.hypot(b[0] - a[0], b[1] - a[1])
      const t0 = Math.atan2(b[1] - a[1], b[0] - a[0])
      let t1 = Math.atan2(h[1] - a[1], h[0] - a[0])
      if (t1 <= t0) t1 += 2 * Math.PI
      return [ring(a, r, 48, t0, t1)]
    }
    if (tool === 'slot') return b ? [[a, b], [b, h]] : [[a, h]]
    return []
  }

  // ---------- pointer ----------

  function startPan(e: React.PointerEvent) {
    gesture.current = { kind: 'pan', x: e.clientX, y: e.clientY, view: v }
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }

  function startDrag(e: React.PointerEvent, drag: (p: XY) => Drag) {
    if (e.button !== 0) return startPan(e)
    e.stopPropagation()
    if (!work) return
    gesture.current = { kind: 'drag', x: e.clientX, y: e.clientY, base: work, drag, moved: false }
    svgRef.current?.setPointerCapture(e.pointerId)
  }

  function onMove(e: React.PointerEvent) {
    setHover(toMm(...local(e)))
    const g = gesture.current
    if (!g) return
    if (g.kind === 'pan' && g.view) {
      setView({ ...g.view, cx: g.view.cx - (e.clientX - g.x) / g.view.k, cy: g.view.cy + (e.clientY - g.y) / g.view.k })
      return
    }
    if (g.kind === 'drag' && g.drag && solver && !solving.current && workRef.current) {
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
      setTimeout(() => workRef.current && commit(workRef.current, before), 30)
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
    if (tool === 'dimension') return dimPick(id)
    setSel((cur) => {
      const n = new Set(e.shiftKey || e.metaKey || e.ctrlKey ? cur : [])
      if (n.has(id) && (e.shiftKey || e.metaKey || e.ctrlKey)) n.delete(id)
      else n.add(id)
      return n
    })
  }

  // The dimension tool: one curve, or two items, then type the value.
  function dimPick(id: string) {
    if (!work) return
    const kind = work.entities.find((e) => e.id === id)?.kind
    const chosen = [...sel, id]
    const single = chosen.length === 1 && (kind === 'line' || kind === 'circle' || kind === 'arc')
    if (chosen.length === 1 && !single) { setSel(new Set(chosen)); return }
    const r = addDimension(work, chosen)
    setSel(new Set())
    if (typeof r === 'string') return setMsg(r)
    commit(r.sketch, work, (s) => {
      const c = s.constraints.find((k) => k.id === r.id)
      const dm = c && dimension(s, c)
      if (c && dm) setEdit({ cid: c.id, x: sx(dm.at[0]), y: sy(dm.at[1]) })
    })
  }

  function onCurveDown(e: React.PointerEvent, ent: Entity) {
    if (e.button !== 0) return startPan(e)
    if (tool !== 'select' && tool !== 'split' && tool !== 'dimension') return // drawing: let the click through
    e.stopPropagation()
    if (!work) return
    const at = toMm(...local(e))
    if (tool === 'split') {
      if (ent.kind === 'line') apply((s) => splitLine(s, ent.id, at))
      else setMsg('Split works on lines; pick a straight segment')
      return
    }
    pick(ent.id, e)
    if (tool === 'dimension') return
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
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') return
      if ((e.key === 'Delete' || e.key === 'Backspace') && sel.size) { e.preventDefault(); apply((s) => deleteItems(s, sel)) }
      else if (e.key === 'z' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); undoLast() }
      else if (e.key === 'Escape') {
        if (clicks.length) setClicks([])
        else if (tool !== 'select') setTool('select')
        else if (sel.size) setSel(new Set())
        else onExit()
      }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  })

  // ---------- glyphs and dimensions ----------

  const ends = (e?: Entity) => (e?.kind === 'line' ? [e.p1, e.p2] : e?.kind === 'arc' ? [e.start, e.end] : [])
  const glyphs = useMemo(() => {
    const out: { c: Constraint; at: XY }[] = []
    if (!work) return out
    for (const c of work.constraints) {
      if (!GLYPH[c.kind]) continue
      const [a, b] = c.refs.map((r) => work.entities.find((e) => e.id === r))
      try {
        if (c.kind === 'tangent' || c.kind === 'perpendicular') {
          const shared = ends(a).find((p) => ends(b).includes(p))
          out.push({ c, at: shared ? pointOf(work, shared) : midpoint(work, a!) })
        } else if (c.kind === 'parallel' || c.kind === 'equal') out.push({ c, at: midpoint(work, a!) }, { c, at: midpoint(work, b!) })
        else if (c.kind === 'horizontal' || c.kind === 'vertical') out.push({ c, at: midpoint(work, a!) })
        else if (c.kind === 'concentric') out.push({ c, at: pointOf(work, (a as { c: string }).c) })
        else out.push({ c, at: a?.kind === 'point' ? [a.x, a.y] : midpoint(work, a!) })
      } catch { /* a stale reference */ }
    }
    return out
  }, [work]) // eslint-disable-line react-hooks/exhaustive-deps

  function dimension(s: Sketch, c: Constraint): { line?: [XY, XY]; at: XY; text: string } | null {
    try {
      const [a, b] = c.refs.map((r) => s.entities.find((e) => e.id === r)!)
      const f = (t: string) => (c.expr ? `${t} = ${c.expr}` : t)
      const paren = (t: string) => (c.driving === false ? `(${t})` : t)
      const n = (x: number) => String(Math.round(x * 100) / 100)
      if (c.kind === 'radius' || c.kind === 'diameter') {
        const centre = pointOf(s, (a as { c: string }).c)
        const m = a.kind === 'arc' ? midpoint(s, a) : ([centre[0] + (a as { r: number }).r * Math.SQRT1_2, centre[1] + (a as { r: number }).r * Math.SQRT1_2] as XY)
        const out: XY = [m[0] + (m[0] - centre[0]) * 0.25, m[1] + (m[1] - centre[1]) * 0.25]
        return { line: [centre, m], at: out, text: paren(f(`${c.kind === 'radius' ? 'R' : 'Ø'}${n(c.value ?? 0)}`)) }
      }
      if (c.kind === 'distance') {
        const p = pointOf(s, a.id)
        let q: XY
        if (b.kind === 'line') {
          const q1 = pointOf(s, b.p1), q2 = pointOf(s, b.p2)
          const vx = q2[0] - q1[0], vy = q2[1] - q1[1]
          const t = ((p[0] - q1[0]) * vx + (p[1] - q1[1]) * vy) / (vx * vx + vy * vy)
          q = [q1[0] + t * vx, q1[1] + t * vy]
        } else q = pointOf(s, b.id)
        return { line: [p, q], at: [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], text: paren(f(n(c.value ?? 0))) }
      }
      if (c.kind === 'angle') return { at: midpoint(s, a), text: paren(f(`${n(((c.value ?? 0) * 180) / Math.PI)}°`)) }
    } catch { /* stale */ }
    return null
  }

  function setDim(cid: string, v: number | string) {
    apply((s) => ({
      ...s,
      constraints: s.constraints.map((c) => {
        if (c.id !== cid) return c
        if (typeof v === 'string') return { ...c, expr: v }
        return { ...c, expr: null, value: c.kind === 'angle' ? (v * Math.PI) / 180 : v }
      }),
    }))
    setEdit(null)
  }

  // ---------- render ----------

  if (!work) return <div className="sketcher"><div className="sk-status">{msg ?? 'Loading sketch…'}</div></div>
  const selIds = [...sel]
  const entity = (id: string) => work.entities.find((e) => e.id === id)
  const options = applicable(work, selIds.filter((id) => entity(id)))
  const curveIds = work.entities.filter((e) => e.kind !== 'point').map((e) => e.id)
  const trouble = new Set([...(report?.conflicting ?? []), ...(report?.redundant ?? [])])
  const troubleRefs = new Set(work.constraints.filter((c) => trouble.has(c.id)).flatMap((c) => c.refs))
  const stacked = new Map<string, number>()
  const drawing = DRAW.find((d) => d.id === tool)
  const editing = edit && work.constraints.find((c) => c.id === edit.cid)

  const toolbar = (
    <div className="toolbar sk" role="toolbar" aria-label="Sketch tools">
      <div className="tgroup"><div className="tbtns">
        {DRAW.map((d) => (
          <button key={d.id} className="tool" aria-pressed={tool === d.id} title={d.hint} onClick={() => { setTool(tool === d.id ? 'select' : d.id); setClicks([]) }}>
            <Icon name={d.icon} size={21} /><span>{d.label}</span>
          </button>
        ))}
        <button className="tool" disabled={!selIds.some((id) => curveIds.includes(id))} onClick={() => apply((s) => toggleConstruction(s, sel))} title="Make the selection reference geometry (or back)">
          <Icon name="construction" size={21} color="var(--muted)" /><span>Construct</span>
        </button>
      </div><div className="tname">Draw</div></div>
      <div className="tgroup"><div className="tbtns">
        <button className="tool" aria-pressed={tool === 'split'} onClick={() => setTool(tool === 'split' ? 'select' : 'split')} title="Click a line to break it in two">
          <Icon name="trimSplit" size={21} /><span>Split</span>
        </button>
        <button className="tool" disabled={sel.size < 2} onClick={() => apply((s) => join(s, selIds.filter((id) => curveIds.includes(id)), 'line'))} title="Join the selected curves into one line">
          <Icon name="line" size={21} /><span>Join</span>
        </button>
        <button className="tool" disabled={selIds.filter((id) => entity(id)?.kind === 'line').length !== 2} title="Round the corner between two selected lines"
          onClick={() => { const r = cornerRadius(work, selIds[0], selIds[1], 3); if (typeof r === 'string') setMsg(r); else { setSel(new Set()); commit(r.sketch, work) } }}>
          <Icon name="fillet" size={21} /><span>Fillet</span>
        </button>
        <button className="tool" disabled={!sel.size} onClick={() => apply((s) => deleteItems(s, sel))}><Icon name="fail" size={21} /><span>Delete</span></button>
        <button className="tool" disabled={!undo.length || busy} onClick={undoLast} title="Undo (Ctrl/⌘ Z)"><Icon name="undo" size={21} /><span>Undo</span></button>
      </div><div className="tname">Modify</div></div>
      <div className="tgroup"><div className="tbtns">
        <button className="tool" aria-pressed={tool === 'dimension'} onClick={() => { setTool(tool === 'dimension' ? 'select' : 'dimension'); setSel(new Set()) }}
          title="Dimension: click a line, circle or arc (or two items); type a number, a parameter or a formula">
          <Icon name="dimension" size={21} color="var(--constraint)" /><span>Dimension</span>
        </button>
        {options.filter((o) => !o.needsValue).map((o) => (
          <button key={o.kind + o.label} className="tool" onClick={() => apply((s) => addConstraint(s, o.kind, selIds))}>
            <span className="glyph">{GLYPH[o.kind] ?? '·'}</span><span>{o.label}</span>
          </button>
        ))}
        {!options.length && <span className="thint">Select items to see the constraints that fit them</span>}
      </div><div className="tname">Constrain</div></div>
      <div className="tgroup"><div className="tbtns">
        <button className="tool finish" onClick={onExit} title="Back to the part (Esc)"><Icon name="ok" size={21} color="var(--sketch)" width={2.2} /><span>Finish</span></button>
      </div><div className="tname">Sketch</div></div>
    </div>
  )

  const canvas = (
    <div className="sketcher" onContextMenu={(e) => e.preventDefault()}>
      <svg ref={svgRef} width={size.w} height={size.h} className={`${tool}${clicks.length ? ' active' : ''}`}
        onPointerDown={(e) => {
          if (e.button !== 0) return startPan(e)
          if (drawing) drawClick(toMm(...local(e)))
          else if (tool === 'select') setSel(new Set())
        }}
        onDoubleClick={() => { if (tool === 'line') setClicks([]) }}
        onPointerMove={onMove} onPointerUp={onUp} onWheel={onWheel} onPointerLeave={() => setHover(null)}>
        <defs>
          <pattern id="grid" width={10 * v.k} height={10 * v.k} patternUnits="userSpaceOnUse" x={sx(0)} y={sy(0)}>
            <path d={`M${10 * v.k} 0H0V${10 * v.k}`} fill="none" className="sk-grid" />
          </pattern>
        </defs>
        {v.k * 10 > 6 && <rect width={size.w} height={size.h} fill="url(#grid)" />}
        <line x1={0} x2={size.w} y1={sy(0)} y2={sy(0)} className="sk-axis x" />
        <line y1={0} y2={size.h} x1={sx(0)} x2={sx(0)} className="sk-axis y" />
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
        {preview().map((pl, i) => <path key={i} d={path(pl)} className="sk-preview" />)}
        {work.constraints.filter((c) => DIMS.has(c.kind)).map((c) => {
          const dm = dimension(work, c)
          if (!dm) return null
          return (
            <g key={c.id} className={`sk-dim${sel.has(c.id) ? ' sel' : ''}${trouble.has(c.id) ? ' bad' : ''}${c.expr ? ' driven' : ''}`}>
              {dm.line && <line x1={sx(dm.line[0][0])} y1={sy(dm.line[0][1])} x2={sx(dm.line[1][0])} y2={sy(dm.line[1][1])} />}
              <text x={sx(dm.at[0])} y={sy(dm.at[1])} onPointerDown={(e) => { e.stopPropagation(); setEdit({ cid: c.id, x: sx(dm.at[0]), y: sy(dm.at[1]) }) }}>{dm.text}</text>
            </g>
          )
        })}
        {glyphs.map(({ c, at }, i) => {
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
          const x = sx(p.x), y = sy(p.y)
          return (
            <rect key={p.id} x={x - 3.5} y={y - 3.5} width={7} height={7} className={`sk-pt${sel.has(p.id) ? ' sel' : ''}${p.fixed ? ' fixed' : ''}`}
              onPointerDown={(e) => {
                if (drawing) return // let the click draw (it snaps to this point)
                pick(p.id, e)
                if (tool === 'select') startDrag(e, (q) => ({ points: [{ id: p.id, x: q[0], y: q[1] }] }))
                else e.stopPropagation()
              }} />
          )
        })}
        {hover && drawing && (() => { const s = snapped(hover); return s.id ? <circle cx={sx(s.at[0])} cy={sy(s.at[1])} r={7} className="sk-snap" /> : null })()}
      </svg>
      {editing && (
        <div className="sk-dimbox" style={{ left: Math.min(edit.x - 20, size.w - 250), top: Math.min(edit.y + 12, size.h - 140) }}>
          <ExprField id="sk-dim" label={NAME[editing.kind]} autoFocus params={params}
            value={editing.kind === 'angle' ? ((editing.value ?? 0) * 180) / Math.PI : editing.value ?? 0}
            expr={editing.expr ?? undefined} unit={editing.kind === 'angle' ? '°' : 'mm'}
            onCommit={(x) => setDim(editing.id, x)} />
          <div className="row"><span className="nm">Enter to apply · a parameter name or formula links it</span><span className="spacer" />
            <button className="link" onMouseDown={(e) => { e.preventDefault(); setEdit(null) }}>Close</button></div>
        </div>
      )}
      <div className="sk-hintbar">{drawing ? drawing.hint : tool === 'dimension' ? 'Click a line, circle or arc, or two items, to dimension them.' : tool === 'split' ? 'Click a line to split it.' : 'Drag to move; click to select (Shift adds). Right or middle drag pans, the wheel zooms.'}
        {tool === 'polygon' && <label className="sides">Sides <input type="number" min={3} max={64} value={sides} onChange={(e) => setSides(Number(e.target.value) || 6)} /></label>}
      </div>
      <div className={`sk-status${report && (report.conflicting.length || report.redundant.length || report.status === 'failed') ? ' bad' : report?.dof === 0 ? ' good' : ''}`}>
        {busy ? 'Solving…' : statusText(report)}{msg && <span className="no"> · {msg}</span>}
      </div>
    </div>
  )

  const sketchTree = (
    <div className="sktree">
      <h2>Sketch {work.id}</h2>
      <div className="xfield"><label htmlFor="sk-name">Name</label>
        <input id="sk-name" defaultValue={work.name} key={work.name} onBlur={(e) => e.target.value !== work.name && apply((s) => ({ ...s, name: e.target.value }))} />
      </div>
      <p className="note">{tree.sketches.find((s) => s.id === sid)?.placement.kind === 'face' ? 'On a face of the part' : `On the ${tree.sketches.find((s) => s.id === sid)?.placement.plane} plane`} · {work.entities.filter((e) => e.kind !== 'point').length} curves</p>
      <div className="secl">Constraints ({work.constraints.length})</div>
      <ul className="clist">
        {work.constraints.map((c) => (
          <li key={c.id} className={`${sel.has(c.id) ? 'sel' : ''}${trouble.has(c.id) ? ' bad' : ''}`}>
            <button onClick={() => setSel(new Set([c.id]))}>
              <span className="g">{GLYPH[c.kind] ?? (c.kind === 'radius' ? 'R' : c.kind === 'diameter' ? 'Ø' : c.kind === 'angle' ? '∠' : '↔')}</span>
              <span className="lb">{NAME[c.kind]}</span>
              {c.expr ? <span className="formula">{c.expr}</span> : DIMS.has(c.kind) ? <span className="nm">{c.kind === 'angle' ? `${Math.round(((c.value ?? 0) * 180) / Math.PI * 100) / 100}°` : Math.round((c.value ?? 0) * 100) / 100}</span> : null}
            </button>
            <button className="x" aria-label={`Delete ${c.id}`} onClick={() => apply((s) => deleteItems(s, new Set([c.id])))}>×</button>
          </li>
        ))}
      </ul>
      {trouble.size > 0 && <p className="note wn">Red rows are conflicting or redundant: delete one to free the sketch.</p>}
      <div className="secl">Parameters it uses</div>
      <p className="note">{tree.sketches.find((s) => s.id === sid)?.inputs.join(', ') || 'none'}</p>
    </div>
  )

  return (
    <>
      {canvas}
      {toolbarHost && createPortal(toolbar, toolbarHost)}
      {treeHost && createPortal(sketchTree, treeHost)}
    </>
  )
}

function statusText(r: SolveReport | null) {
  if (!r) return 'Empty sketch: pick a tool above and draw'
  if (r.status === 'failed') return 'The solver could not satisfy every constraint'
  const parts = [r.dof === 0 ? 'Fully constrained' : `${r.dof} degree${r.dof === 1 ? '' : 's'} of freedom`]
  if (r.conflicting.length) parts.push(`conflicting: ${r.conflicting.join(', ')}`)
  if (r.redundant.length) parts.push(`redundant: ${r.redundant.join(', ')}`)
  return parts.join(' · ')
}
