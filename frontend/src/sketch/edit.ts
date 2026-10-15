// Pure edits on a sketch: each returns a new sketch (the editor keeps the old ones for undo).
import { pointOf, type XY } from './draw.ts'
import type { Constraint, ConstraintKind, Entity, Sketch } from './model.ts'

export function nextId(s: Sketch, prefix: string): string {
  const used = new Set([...s.entities.map((e) => e.id), ...s.constraints.map((c) => c.id)])
  let n = 1
  while (used.has(`${prefix}${n}`)) n++
  return `${prefix}${n}`
}

const refsOf = (e: Entity): string[] =>
  e.kind === 'line' ? [e.p1, e.p2] : e.kind === 'circle' ? [e.c] : e.kind === 'arc' ? [e.c, e.start, e.end] : e.kind === 'spline' ? e.points : []

/** Drop points nothing uses any more (pin points always stay). */
function prune(s: Sketch): Sketch {
  const used = new Set(s.entities.flatMap(refsOf))
  s.constraints.forEach((c) => c.refs.forEach((r) => used.add(r)))
  return { ...s, entities: s.entities.filter((e) => e.kind !== 'point' || e.ref || used.has(e.id)) }
}

export function deleteItems(s: Sketch, ids: Set<string>): Sketch {
  // Deleting a point deletes the curves on it; deleting anything deletes the constraints on it.
  const gone = new Set(ids)
  for (const e of s.entities) if (refsOf(e).some((r) => gone.has(r))) gone.add(e.id)
  for (const e of s.entities) if (e.kind === 'point' && e.ref) gone.delete(e.id) // pins are the intent's, not the sketch's
  return prune({
    ...s,
    entities: s.entities.filter((e) => !gone.has(e.id)),
    constraints: s.constraints.filter((c) => !gone.has(c.id) && !c.refs.some((r) => gone.has(r))),
  })
}

export function toggleConstruction(s: Sketch, ids: Set<string>): Sketch {
  return { ...s, entities: s.entities.map((e) => (ids.has(e.id) && e.kind !== 'point' ? { ...e, construction: !e.construction } : e)) }
}

/** Break a line in two at `at` (projected onto it). The new point is free, so it can be dragged off the line. */
export function splitLine(s: Sketch, lineId: string, at: XY): Sketch {
  const l = s.entities.find((e) => e.id === lineId)
  if (!l || l.kind !== 'line') return s
  const a = pointOf(s, l.p1), b = pointOf(s, l.p2)
  const vx = b[0] - a[0], vy = b[1] - a[1]
  const t = Math.max(0.05, Math.min(0.95, ((at[0] - a[0]) * vx + (at[1] - a[1]) * vy) / (vx * vx + vy * vy)))
  const pid = nextId(s, 'p')
  const s1: Sketch = { ...s, entities: [...s.entities, { id: pid, kind: 'point', x: a[0] + t * vx, y: a[1] + t * vy }] }
  const l2 = nextId(s1, 'l')
  const entities = s1.entities.flatMap((e): Entity[] =>
    e.id === lineId ? [{ ...l, p2: pid }, { id: l2, kind: 'line', p1: pid, p2: l.p2, construction: l.construction }] : [e])
  // Constraints on the old line: endpoint tangency follows its end; length dimensions go (they no longer mean the same).
  const constraints = s.constraints.flatMap((c): Constraint[] => {
    if (!c.refs.includes(lineId)) return [c]
    if (c.kind === 'distance') return []
    if (c.kind === 'tangent') {
      const other = c.refs.find((r) => r !== lineId)!
      const oe = s.entities.find((e) => e.id === other)
      const ends = oe?.kind === 'line' ? [oe.p1, oe.p2] : oe?.kind === 'arc' ? [oe.start, oe.end] : []
      return [{ ...c, refs: c.refs.map((r) => (r === lineId ? (ends.includes(l.p2) ? l2 : lineId) : r)) }]
    }
    if (c.kind === 'horizontal' || c.kind === 'vertical') return [c, { ...c, id: `${c.id}b`, refs: [l2] }]
    return [c]
  })
  return { ...s, entities, constraints }
}

/** Connected curves in order: [entity ids], [point ids along the chain], or null if they do not form one chain. */
function chain(s: Sketch, ids: string[]): { order: string[]; pts: string[] } | null {
  const ends = (id: string): [string, string] | null => {
    const e = s.entities.find((x) => x.id === id)
    if (e?.kind === 'line') return [e.p1, e.p2]
    if (e?.kind === 'arc') return [e.start, e.end]
    if (e?.kind === 'spline') return [e.points[0], e.points[e.points.length - 1]]
    return null
  }
  const segs = ids.map((id) => [id, ends(id)] as const)
  if (segs.some(([, e]) => !e)) return null
  const count = new Map<string, number>()
  segs.forEach(([, e]) => e!.forEach((p) => count.set(p, (count.get(p) ?? 0) + 1)))
  const start = [...count.entries()].find(([, n]) => n === 1)?.[0]
  if (!start) return null
  const order: string[] = [], pts = [start]
  const left = new Map(segs.map(([id, e]) => [id, e!]))
  let cur = start
  while (left.size) {
    const hit = [...left.entries()].find(([, e]) => e.includes(cur))
    if (!hit) return null
    const [id, e] = hit
    left.delete(id)
    order.push(id)
    cur = e[0] === cur ? e[1] : e[0]
    pts.push(cur)
  }
  return { order, pts }
}

/** Replace a chain of connected curves with one line (its two ends) or one spline (through every joint, plus the middle
 *  of each arc so the shape survives). Constraints on the removed curves go, except tangency at the chain's ends. */
export function join(s: Sketch, ids: string[], to: 'line' | 'spline'): Sketch | string {
  const ch = chain(s, ids)
  if (!ch || ch.order.length < 2) return 'Select two or more curves joined end to end'
  const first = ch.pts[0], last = ch.pts[ch.pts.length - 1]
  if (first === last) return 'A closed loop cannot be joined into one curve; leave one segment out'
  let work = s
  let through: string[] = [first]
  if (to === 'spline') {
    for (let i = 0; i < ch.order.length; i++) {
      const e = s.entities.find((x) => x.id === ch.order[i])!
      if (e.kind === 'arc') {
        // Keep the arc's bulge: a point at its middle.
        const c = pointOf(s, e.c), a = pointOf(s, e.start), b = pointOf(s, e.end)
        let a0 = Math.atan2(a[1] - c[1], a[0] - c[0]), a1 = Math.atan2(b[1] - c[1], b[0] - c[0])
        if (a1 <= a0) a1 += 2 * Math.PI
        const r = Math.hypot(a[0] - c[0], a[1] - c[1]), tm = (a0 + a1) / 2
        const mid = nextId(work, 'p')
        work = { ...work, entities: [...work.entities, { id: mid, kind: 'point', x: c[0] + r * Math.cos(tm), y: c[1] + r * Math.sin(tm) }] }
        through.push(mid)
      } else if (e.kind === 'spline') {
        const inner = ch.pts[i] === e.points[0] ? e.points.slice(1, -1) : [...e.points].reverse().slice(1, -1)
        through.push(...inner)
      }
      through.push(ch.pts[i + 1])
    }
  } else through = [first, last]
  const nid = nextId(work, to === 'line' ? 'l' : 's')
  const removed = new Set(ch.order)
  const keepTangent = (c: Constraint) => {
    if (c.kind !== 'tangent' || to === 'spline') return null
    const mine = c.refs.find((r) => removed.has(r))
    const other = c.refs.find((r) => !removed.has(r))
    const oe = work.entities.find((e) => e.id === other)
    const ends = oe?.kind === 'line' ? [oe.p1, oe.p2] : oe?.kind === 'arc' ? [oe.start, oe.end] : []
    return mine && other && (ends.includes(first) || ends.includes(last)) ? { ...c, refs: [nid, other] } : null
  }
  const constraints = work.constraints.flatMap((c) => {
    if (!c.refs.some((r) => removed.has(r))) return [c]
    const t = keepTangent(c)
    return t ? [t] : []
  })
  // A spline keeps the chain's smoothness where it was tangent to its neighbours at either end.
  const tangentAt = (p: string) => work.constraints.some((c) => c.kind === 'tangent' && c.refs.some((r) => removed.has(r)) &&
    c.refs.some((r) => { const o = work.entities.find((e) => e.id === r && !removed.has(r)); return (o?.kind === 'line' && (o.p1 === p || o.p2 === p)) || (o?.kind === 'arc' && (o.start === p || o.end === p)) }))
  const newEntity: Entity = to === 'line' ? { id: nid, kind: 'line', p1: first, p2: last }
    : { id: nid, kind: 'spline', points: through, tangent_start: tangentAt(first), tangent_end: tangentAt(last) }
  return prune({ ...work, entities: [...work.entities.filter((e) => !removed.has(e.id)), newEntity], constraints })
}

export interface ConstraintOption {
  kind: ConstraintKind
  label: string
  needsValue?: boolean
}

/** Which constraints make sense for the selection (as CATIA's Constraints Defined in Dialog Box offers them). */
export function applicable(s: Sketch, ids: string[]): ConstraintOption[] {
  const kinds = ids.map((id) => s.entities.find((e) => e.id === id)?.kind)
  const n = (k: string) => kinds.filter((x) => x === k).length
  const curves = n('circle') + n('arc')
  const out: ConstraintOption[] = []
  if (ids.length === 1 && n('line') === 1) out.push({ kind: 'horizontal', label: 'Horizontal' }, { kind: 'vertical', label: 'Vertical' }, { kind: 'distance', label: 'Length', needsValue: true })
  if (ids.length === 1 && curves === 1) out.push({ kind: 'radius', label: 'Radius', needsValue: true }, { kind: 'diameter', label: 'Diameter', needsValue: true })
  if (ids.length === 1 && n('point') === 1) out.push({ kind: 'fix', label: 'Fix' })
  if (ids.length === 2 && n('line') === 2) out.push({ kind: 'parallel', label: 'Parallel' }, { kind: 'perpendicular', label: 'Perpendicular' }, { kind: 'equal', label: 'Equal length' }, { kind: 'angle', label: 'Angle', needsValue: true })
  if (ids.length === 2 && n('line') === 1 && curves === 1) out.push({ kind: 'tangent', label: 'Tangent' })
  if (ids.length === 2 && curves === 2) out.push({ kind: 'tangent', label: 'Tangent' }, { kind: 'concentric', label: 'Concentric' }, { kind: 'equal', label: 'Equal radius' })
  if (ids.length === 2 && n('point') === 2) out.push({ kind: 'coincident', label: 'Coincident' }, { kind: 'distance', label: 'Distance', needsValue: true })
  if (ids.length === 2 && n('point') === 1 && n('line') === 1) out.push({ kind: 'point_on', label: 'On line' }, { kind: 'distance', label: 'Distance', needsValue: true })
  if (ids.length === 2 && n('point') === 1 && curves === 1) out.push({ kind: 'point_on', label: 'On curve' })
  return out
}

/** Add a constraint, measuring its current value for a dimension (so adding one never jumps the geometry). */
export function addConstraint(s: Sketch, kind: ConstraintKind, ids: string[]): Sketch {
  // Points first for point/curve pairs, as the solver expects.
  const ordered = [...ids].sort((a, b) => Number(s.entities.find((e) => e.id === b)?.kind === 'point') - Number(s.entities.find((e) => e.id === a)?.kind === 'point'))
  let refs = ordered
  let value: number | undefined
  const ent = (id: string) => s.entities.find((e) => e.id === id)!
  if (kind === 'distance') {
    if (ids.length === 1) {
      const l = ent(ids[0]) as Extract<Entity, { kind: 'line' }>
      refs = [l.p1, l.p2]
    }
    const [a, b] = refs.map(ent)
    if (b.kind === 'line') {
      const p = pointOf(s, a.id), q1 = pointOf(s, b.p1), q2 = pointOf(s, b.p2)
      const vx = q2[0] - q1[0], vy = q2[1] - q1[1]
      value = Math.abs((p[0] - q1[0]) * vy - (p[1] - q1[1]) * vx) / Math.hypot(vx, vy)
    } else {
      const p = pointOf(s, a.id), q = pointOf(s, b.id)
      value = Math.hypot(p[0] - q[0], p[1] - q[1])
    }
  } else if (kind === 'radius' || kind === 'diameter') {
    const e = ent(ids[0])
    const r = e.kind === 'circle' ? e.r : e.kind === 'arc' ? Math.hypot(pointOf(s, e.start)[0] - pointOf(s, e.c)[0], pointOf(s, e.start)[1] - pointOf(s, e.c)[1]) : 0
    value = kind === 'radius' ? r : 2 * r
  } else if (kind === 'angle') {
    const [l1, l2] = ids.map(ent) as Extract<Entity, { kind: 'line' }>[]
    const d = (l: typeof l1) => { const a = pointOf(s, l.p1), b = pointOf(s, l.p2); return Math.atan2(b[1] - a[1], b[0] - a[0]) }
    value = d(l2) - d(l1)
  }
  return { ...s, constraints: [...s.constraints, { id: nextId(s, 'k'), kind, refs, value }] }
}

export function setValue(s: Sketch, cid: string, value: number): Sketch {
  return { ...s, constraints: s.constraints.map((c) => (c.id === cid ? { ...c, value } : c)) }
}

/** Corner radius between two lines (CATIA's Corner): both lines are trimmed back from their corner (or extended to it if
 *  they do not meet) and joined by an arc tangent to both, with a radius dimension. Returns the new sketch and the
 *  radius constraint's id, or a message. The radius shrinks to fit if the lines are too short for it. */
export function cornerRadius(s: Sketch, l1Id: string, l2Id: string, r: number): { sketch: Sketch; dim: string } | string {
  const l1 = s.entities.find((e) => e.id === l1Id), l2 = s.entities.find((e) => e.id === l2Id)
  if (l1?.kind !== 'line' || l2?.kind !== 'line') return 'Select two lines'
  const P = (id: string) => pointOf(s, id)
  // The corner: a shared point, else where the two lines (extended) cross.
  const shared = [l1.p1, l1.p2].find((p) => p === l2.p1 || p === l2.p2)
  let corner: XY
  let near1: string, near2: string // each line's end at the corner (replaced by the tangent point)
  if (shared) {
    corner = P(shared)
    near1 = near2 = shared
  } else {
    const [a, b] = [P(l1.p1), P(l1.p2)], [c, d] = [P(l2.p1), P(l2.p2)]
    const den = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
    if (Math.abs(den) < 1e-9) return 'The lines are parallel: there is no corner to round'
    const t = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / den
    corner = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
    const dist = (p: XY) => Math.hypot(p[0] - corner[0], p[1] - corner[1])
    near1 = dist(a) < dist(b) ? l1.p1 : l1.p2
    near2 = dist(c) < dist(d) ? l2.p1 : l2.p2
  }
  const far1 = near1 === l1.p1 ? l1.p2 : l1.p1, far2 = near2 === l2.p1 ? l2.p2 : l2.p1
  const unit = (p: XY): [XY, number] => { const v: XY = [p[0] - corner[0], p[1] - corner[1]]; const n = Math.hypot(...v); return [[v[0] / n, v[1] / n], n] }
  const [u1, len1] = unit(P(far1)), [u2, len2] = unit(P(far2))
  const theta = Math.acos(Math.max(-1, Math.min(1, u1[0] * u2[0] + u1[1] * u2[1])))
  if (theta < 1e-3 || Math.PI - theta < 1e-3) return 'The lines are in line: there is no corner to round'
  let rr = r, d = rr / Math.tan(theta / 2)
  if (d > 0.9 * Math.min(len1, len2)) { d = 0.9 * Math.min(len1, len2); rr = d * Math.tan(theta / 2) }
  const t1: XY = [corner[0] + u1[0] * d, corner[1] + u1[1] * d], t2: XY = [corner[0] + u2[0] * d, corner[1] + u2[1] * d]
  const bis: XY = [(u1[0] + u2[0]), (u1[1] + u2[1])]
  const bn = Math.hypot(...bis), h = rr / Math.sin(theta / 2)
  const centre: XY = [corner[0] + (bis[0] / bn) * h, corner[1] + (bis[1] / bn) * h]
  // New points: the two tangent points and the centre; the arc runs counter-clockwise between them.
  let w = s
  const add = (x: XY) => { const id = nextId(w, 'p'); w = { ...w, entities: [...w.entities, { id, kind: 'point', x: x[0], y: x[1] }] }; return id }
  const pt1 = add(t1), pt2 = add(t2), pc = add(centre)
  const ccw = (t1[0] - centre[0]) * (t2[1] - centre[1]) - (t1[1] - centre[1]) * (t2[0] - centre[0]) > 0
  const aid = nextId(w, 'a')
  const arc: Entity = { id: aid, kind: 'arc', c: pc, start: ccw ? pt1 : pt2, end: ccw ? pt2 : pt1 }
  const entities = w.entities.map((e): Entity => {
    if (e.id === l1Id && e.kind === 'line') return e.p1 === near1 ? { ...e, p1: pt1 } : { ...e, p2: pt1 }
    if (e.id === l2Id && e.kind === 'line') return e.p1 === near2 ? { ...e, p1: pt2 } : { ...e, p2: pt2 }
    return e
  })
  // The old corner point is no longer on either line: constraints that held it go with it (if nothing else uses it).
  const corners = new Set([near1, near2])
  const stillUsed = (p: string) => entities.some((e) => e.kind !== 'point' && (refsOf(e) as string[]).includes(p))
  const dropped = new Set([...corners].filter((p) => !stillUsed(p)))
  let w2: Sketch = { ...w, entities: [...entities, arc], constraints: w.constraints.filter((c) => !c.refs.some((x) => dropped.has(x))) }
  const k1 = nextId(w2, 'k')
  w2 = { ...w2, constraints: [...w2.constraints, { id: k1, kind: 'tangent', refs: [l1Id, aid] }] }
  const k2 = nextId(w2, 'k')
  w2 = { ...w2, constraints: [...w2.constraints, { id: k2, kind: 'tangent', refs: [aid, l2Id] }] }
  const k3 = nextId(w2, 'k')
  w2 = { ...w2, constraints: [...w2.constraints, { id: k3, kind: 'radius', refs: [aid], value: rr }] }
  return { sketch: prune(w2), dim: k3 }
}
