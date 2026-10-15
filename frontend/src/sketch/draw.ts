// Sketch curves as polylines (mm), for drawing and hit-testing.
import type { Entity, Sketch } from './model.ts'

export type XY = [number, number]

export function pointOf(s: Sketch, id: string): XY {
  const e = s.entities.find((x) => x.id === id)
  if (!e || e.kind !== 'point') throw new Error(`${id} is not a point`)
  return [e.x, e.y]
}

export function arcAngles(s: Sketch, e: Extract<Entity, { kind: 'arc' }>): { c: XY; r: number; a0: number; a1: number } {
  const c = pointOf(s, e.c), a = pointOf(s, e.start), b = pointOf(s, e.end)
  const a0 = Math.atan2(a[1] - c[1], a[0] - c[0])
  let a1 = Math.atan2(b[1] - c[1], b[0] - c[0])
  if (a1 <= a0) a1 += 2 * Math.PI
  return { c, r: Math.hypot(a[0] - c[0], a[1] - c[1]), a0, a1 }
}

/** Bezier control points of a Catmull–Rom spline through `pts` (the backend builds the same segments). An end with a
 *  unit tangent leaves along it, a third of the first chord long. */
export function splineControls(pts: XY[], t0?: XY | null, t1?: XY | null): [XY, XY, XY, XY][] {
  const ext = [pts[0], ...pts, pts[pts.length - 1]]
  const segs: [XY, XY, XY, XY][] = []
  for (let i = 1; i < ext.length - 2; i++) {
    const [p0, p1, p2, p3] = [ext[i - 1], ext[i], ext[i + 1], ext[i + 2]]
    const chord = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 3
    let c1: XY = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6]
    let c2: XY = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6]
    if (i === 1 && t0) c1 = [p1[0] + t0[0] * chord, p1[1] + t0[1] * chord]
    if (i === ext.length - 3 && t1) c2 = [p2[0] - t1[0] * chord, p2[1] - t1[1] * chord]
    segs.push([p1, c1, c2, p2])
  }
  return segs
}

/** Unit direction a spline should leave `end` along: the tangent of the other curve on that point, pointing toward
 *  `next` (the spline's neighbouring point). */
export function joinTangent(s: Sketch, splineId: string, end: string, next: string): XY | null {
  const p = pointOf(s, end), q = pointOf(s, next)
  for (const e of s.entities) {
    if (e.id === splineId || ('construction' in e && e.construction)) continue
    let d: XY | null = null
    if (e.kind === 'line' && (e.p1 === end || e.p2 === end)) {
      const a = pointOf(s, e.p1), b = pointOf(s, e.p2)
      d = [b[0] - a[0], b[1] - a[1]]
    } else if (e.kind === 'arc' && (e.start === end || e.end === end)) {
      const c = pointOf(s, e.c)
      d = [-(p[1] - c[1]), p[0] - c[0]]
    }
    if (d) {
      const n = Math.hypot(d[0], d[1]) || 1
      const sign = d[0] * (q[0] - p[0]) + d[1] * (q[1] - p[1]) >= 0 ? 1 : -1
      return [(sign * d[0]) / n, (sign * d[1]) / n]
    }
  }
  return null
}

/** Centripetal Catmull–Rom through the points, sampled. */
export function splinePoints(pts: XY[], per = 16, t0?: XY | null, t1?: XY | null): XY[] {
  if (pts.length < 2) return pts
  if (pts.length === 2 && !t0 && !t1) return pts
  const out: XY[] = []
  const segs = splineControls(pts, t0, t1)
  for (let i = 0; i < segs.length; i++) {
    const [p1, c1, c2, p2] = segs[i]
    for (let k = i === 0 ? 0 : 1; k <= per; k++) {
      const t = k / per, u = 1 - t
      out.push([
        u * u * u * p1[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * p2[0],
        u * u * u * p1[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * p2[1],
      ])
    }
  }
  return out
}

export function polyline(s: Sketch, e: Entity): XY[] {
  switch (e.kind) {
    case 'line': return [pointOf(s, e.p1), pointOf(s, e.p2)]
    case 'circle': {
      const c = pointOf(s, e.c)
      return Array.from({ length: 73 }, (_, i) => [c[0] + e.r * Math.cos((i / 72) * 2 * Math.PI), c[1] + e.r * Math.sin((i / 72) * 2 * Math.PI)] as XY)
    }
    case 'arc': {
      const { c, r, a0, a1 } = arcAngles(s, e)
      const n = Math.max(8, Math.ceil(((a1 - a0) / (2 * Math.PI)) * 72))
      return Array.from({ length: n + 1 }, (_, i) => {
        const t = a0 + ((a1 - a0) * i) / n
        return [c[0] + r * Math.cos(t), c[1] + r * Math.sin(t)] as XY
      })
    }
    case 'spline': {
      const n = e.points.length
      const t0 = e.tangent_start && n > 1 ? joinTangent(s, e.id, e.points[0], e.points[1]) : null
      const t1raw = e.tangent_end && n > 1 ? joinTangent(s, e.id, e.points[n - 1], e.points[n - 2]) : null
      const t1: XY | null = t1raw ? [-t1raw[0], -t1raw[1]] : null // arriving along it, not leaving
      return splinePoints(e.points.map((p) => pointOf(s, p)), 16, t0, t1)
    }
    default: return []
  }
}

/** A point on the curve near its middle, for labels. */
export function midpoint(s: Sketch, e: Entity): XY {
  const pl = polyline(s, e)
  return pl[Math.floor(pl.length / 2)] ?? [0, 0]
}

export function distToPolyline(p: XY, pl: XY[]): { d: number; at: XY } {
  let best = { d: Infinity, at: p }
  for (let i = 0; i + 1 < pl.length; i++) {
    const [a, b] = [pl[i], pl[i + 1]]
    const vx = b[0] - a[0], vy = b[1] - a[1]
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / (vx * vx + vy * vy || 1)))
    const q: XY = [a[0] + t * vx, a[1] + t * vy]
    const d = Math.hypot(p[0] - q[0], p[1] - q[1])
    if (d < best.d) best = { d, at: q }
  }
  return best
}
