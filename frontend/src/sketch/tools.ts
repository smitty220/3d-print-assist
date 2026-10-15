// Drawing tools for the sketcher: each adds geometry to a sketch and the constraints a CAD user expects from it (a
// rectangle stays square-cornered, a polygon stays regular, a near-level line is held horizontal). Pure functions:
// they return a new sketch, so the editor can undo. A point given as `{ id }` reuses an existing point (a snap: the
// new curve joins it); `{ at }` makes a new one.
import { addConstraint, nextId } from './edit'
import type { Constraint, Entity, Sketch } from './model'
import type { XY } from './draw'

export type PointRef = { id: string } | { at: XY }

const SNAP_ANGLE = (3 * Math.PI) / 180

function point(s: Sketch, ref: PointRef): { s: Sketch; id: string; at: XY } {
  if ('id' in ref) {
    const p = s.entities.find((e) => e.id === ref.id)
    if (p?.kind !== 'point') throw new Error(`${ref.id} is not a point`)
    return { s, id: ref.id, at: [p.x, p.y] }
  }
  const id = nextId(s, 'p')
  return { s: { ...s, entities: [...s.entities, { id, kind: 'point', x: ref.at[0], y: ref.at[1] }] }, id, at: ref.at }
}

function add(s: Sketch, e: Entity): Sketch {
  return { ...s, entities: [...s.entities, e] }
}

function con(s: Sketch, c: Omit<Constraint, 'id'>): Sketch {
  return { ...s, constraints: [...s.constraints, { id: nextId(s, 'k'), ...c }] }
}

/** The point of the sketch within `tol` (mm) of `at`, for snapping. */
export function nearestPoint(s: Sketch, at: XY, tol: number): string | null {
  let best: string | null = null
  let d = tol
  for (const e of s.entities) {
    if (e.kind !== 'point') continue
    const dd = Math.hypot(e.x - at[0], e.y - at[1])
    if (dd <= d) { d = dd; best = e.id }
  }
  return best
}

export function addPoint(s: Sketch, at: XY): { sketch: Sketch; id: string } {
  const r = point(s, { at })
  return { sketch: r.s, id: r.id }
}

/** A line between two points, held horizontal or vertical when drawn within 3° of it. */
export function addLine(s: Sketch, a: PointRef, b: PointRef): { sketch: Sketch; id: string; end: string } {
  const p = point(s, a)
  const q = point(p.s, b)
  if (p.id === q.id) throw new Error('a line needs two different points')
  const id = nextId(q.s, 'l')
  let out = add(q.s, { id, kind: 'line', p1: p.id, p2: q.id })
  const ang = Math.atan2(q.at[1] - p.at[1], q.at[0] - p.at[0])
  const fromH = Math.min(Math.abs(ang), Math.abs(Math.abs(ang) - Math.PI))
  const fromV = Math.abs(Math.abs(ang) - Math.PI / 2)
  if (fromH < SNAP_ANGLE) out = con(out, { kind: 'horizontal', refs: [id] })
  else if (fromV < SNAP_ANGLE) out = con(out, { kind: 'vertical', refs: [id] })
  return { sketch: out, id, end: q.id }
}

/** A rectangle from two opposite corners: four lines, two horizontal and two vertical. */
export function addRectangle(s: Sketch, a: XY, b: XY): { sketch: Sketch; lines: string[] } {
  if (Math.abs(a[0] - b[0]) < 1e-6 || Math.abs(a[1] - b[1]) < 1e-6) throw new Error('a rectangle needs width and height')
  const corners: XY[] = [a, [b[0], a[1]], b, [a[0], b[1]]]
  let out = s
  const ids: string[] = []
  for (const c of corners) {
    const r = point(out, { at: c })
    out = r.s
    ids.push(r.id)
  }
  const lines: string[] = []
  for (let i = 0; i < 4; i++) {
    const id = nextId(out, 'l')
    out = add(out, { id, kind: 'line', p1: ids[i], p2: ids[(i + 1) % 4] })
    out = con(out, { kind: i % 2 === 0 ? 'horizontal' : 'vertical', refs: [id] })
    lines.push(id)
  }
  return { sketch: out, lines }
}

/** A rectangle centred on a point: also a construction diagonal whose midpoint is the centre. */
export function addCenterRectangle(s: Sketch, centre: XY, corner: XY): { sketch: Sketch; lines: string[] } {
  const a: XY = [2 * centre[0] - corner[0], 2 * centre[1] - corner[1]]
  const r = addRectangle(s, a, corner)
  let out = r.sketch
  const l0 = out.entities.find((e) => e.id === r.lines[0]) as Extract<Entity, { kind: 'line' }>
  const l1 = out.entities.find((e) => e.id === r.lines[1]) as Extract<Entity, { kind: 'line' }>
  const c = point(out, { at: centre })
  out = c.s
  const diag = nextId(out, 'l')
  out = add(out, { id: diag, kind: 'line', p1: l0.p1, p2: l1.p2, construction: true })
  out = con(out, { kind: 'point_on', refs: [c.id, diag] })
  // The centre stays mid-diagonal: equal distances to both ends.
  const d = Math.hypot(corner[0] - centre[0], corner[1] - centre[1])
  out = con(out, { kind: 'distance', refs: [c.id, l0.p1], value: d, driving: true })
  out = con(out, { kind: 'distance', refs: [c.id, l1.p2], value: d, driving: true })
  return { sketch: out, lines: r.lines }
}

export function addCircle(s: Sketch, centre: PointRef, r: number): { sketch: Sketch; id: string } {
  if (!(r > 0)) throw new Error('a circle needs a radius')
  const c = point(s, centre)
  const id = nextId(c.s, 'k')
  return { sketch: add(c.s, { id, kind: 'circle', c: c.id, r }), id }
}

/** An arc counter-clockwise from `start` to `end` about `centre`; the end is moved onto the start's radius. */
export function addArc(s: Sketch, centre: PointRef, start: PointRef, end: XY): { sketch: Sketch; id: string } {
  const c = point(s, centre)
  const a = point(c.s, start)
  const r = Math.hypot(a.at[0] - c.at[0], a.at[1] - c.at[1])
  if (!(r > 0)) throw new Error('an arc needs a radius')
  const t = Math.atan2(end[1] - c.at[1], end[0] - c.at[0])
  const b = point(a.s, { at: [c.at[0] + r * Math.cos(t), c.at[1] + r * Math.sin(t)] })
  const id = nextId(b.s, 'a')
  return { sketch: add(b.s, { id, kind: 'arc', c: c.id, start: a.id, end: b.id }), id }
}

/** A regular polygon: its corners on a construction circle, every side equal, the first side level. */
export function addPolygon(s: Sketch, centre: XY, corner: XY, sides: number): { sketch: Sketch; circle: string; lines: string[] } {
  const n = Math.max(3, Math.min(64, Math.round(sides)))
  const r = Math.hypot(corner[0] - centre[0], corner[1] - centre[1])
  if (!(r > 0)) throw new Error('a polygon needs a size')
  const c = point(s, { at: centre })
  let out = c.s
  const circle = nextId(out, 'k')
  out = add(out, { id: circle, kind: 'circle', c: c.id, r, construction: true })
  // Start so the first side is level (a flat base): corners at -90° ± 180°/n.
  const t0 = -Math.PI / 2 - Math.PI / n
  const pts: string[] = []
  for (let i = 0; i < n; i++) {
    const t = t0 + (2 * Math.PI * i) / n
    const p = point(out, { at: [centre[0] + r * Math.cos(t), centre[1] + r * Math.sin(t)] })
    out = con(p.s, { kind: 'point_on', refs: [p.id, circle] })
    pts.push(p.id)
  }
  const lines: string[] = []
  for (let i = 0; i < n; i++) {
    const id = nextId(out, 'l')
    out = add(out, { id, kind: 'line', p1: pts[i], p2: pts[(i + 1) % n] })
    lines.push(id)
  }
  for (let i = 1; i < n; i++) out = con(out, { kind: 'equal', refs: [lines[0], lines[i]] })
  out = con(out, { kind: 'horizontal', refs: [lines[0]] })
  return { sketch: out, circle, lines }
}

/** A straight slot (a stadium) between two centres, `width` wide: two lines and two half-circle arcs, tangent. */
export function addSlot(s: Sketch, a: XY, b: XY, width: number): { sketch: Sketch } {
  const L = Math.hypot(b[0] - a[0], b[1] - a[1])
  if (!(L > 0) || !(width > 0)) throw new Error('a slot needs a length and a width')
  const ux = (b[0] - a[0]) / L, uy = (b[1] - a[1]) / L
  const nx = -uy * (width / 2), ny = ux * (width / 2)
  let out = s
  const P = (at: XY) => { const r = point(out, { at }); out = r.s; return r.id }
  const ca = P(a), cb = P(b)
  const a1 = P([a[0] + nx, a[1] + ny]), a2 = P([a[0] - nx, a[1] - ny])
  const b1 = P([b[0] + nx, b[1] + ny]), b2 = P([b[0] - nx, b[1] - ny])
  const l1 = nextId(out, 'l'); out = add(out, { id: l1, kind: 'line', p1: a2, p2: b2 })
  const l2 = nextId(out, 'l'); out = add(out, { id: l2, kind: 'line', p1: b1, p2: a1 })
  const arcB = nextId(out, 'a'); out = add(out, { id: arcB, kind: 'arc', c: cb, start: b2, end: b1 })
  const arcA = nextId(out, 'a'); out = add(out, { id: arcA, kind: 'arc', c: ca, start: a1, end: a2 })
  out = con(out, { kind: 'tangent', refs: [l1, arcB] })
  out = con(out, { kind: 'tangent', refs: [l2, arcB] })
  out = con(out, { kind: 'tangent', refs: [l2, arcA] })
  out = con(out, { kind: 'tangent', refs: [l1, arcA] })
  out = con(out, { kind: 'equal', refs: [arcA, arcB] })
  return { sketch: out }
}

/** The dimension that fits a selection: a line's length, a circle's diameter, an arc's radius, two points' distance,
 *  a point to a line, two lines' angle. Measures the current geometry, so adding it never moves anything. */
export function addDimension(s: Sketch, ids: string[]): { sketch: Sketch; id: string } | string {
  const kinds = ids.map((id) => s.entities.find((e) => e.id === id)?.kind)
  let kind: Constraint['kind'] | null = null
  if (ids.length === 1 && kinds[0] === 'line') kind = 'distance'
  else if (ids.length === 1 && kinds[0] === 'circle') kind = 'diameter'
  else if (ids.length === 1 && kinds[0] === 'arc') kind = 'radius'
  else if (ids.length === 2 && kinds.every((k) => k === 'point')) kind = 'distance'
  else if (ids.length === 2 && kinds.includes('point') && kinds.includes('line')) kind = 'distance'
  else if (ids.length === 2 && kinds.every((k) => k === 'line')) kind = 'angle'
  if (!kind) return 'Pick a line, circle or arc, or two points, a point and a line, or two lines'
  const out = addConstraint(s, kind, ids)
  return { sketch: out, id: out.constraints[out.constraints.length - 1].id }
}
