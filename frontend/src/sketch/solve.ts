// Solve a sketch with planegcs (FreeCAD's sketch solver, as WebAssembly). The same file runs in the browser and, via
// Node, behind the backend (frontend/sketch-cli.ts), so both always agree.
import { GcsWrapper, init_planegcs_module } from '@salusoft89/planegcs'
import type { Constraint, Entity, Sketch, SolveReport } from './model.ts'

type Prim = Record<string, unknown> & { id: string; type: string }
type Module = Awaited<ReturnType<typeof init_planegcs_module>>

let modPromise: Promise<Module> | null = null

/** Load the solver once. In the browser pass `locateFile` for the .wasm URL. */
export function loadSolver(options?: Parameters<typeof init_planegcs_module>[0]): Promise<Module> {
  const p = modPromise ?? init_planegcs_module(options)
  modPromise = p
  return p
}

const pt = (s: Sketch, id: string) => {
  const e = s.entities.find((x) => x.id === id)
  if (!e || e.kind !== 'point') throw new Error(`${id} is not a point`)
  return e
}

function toPrims(s: Sketch): Prim[] {
  const out: Prim[] = []
  const kind = new Map(s.entities.map((e) => [e.id, e.kind]))
  // Points first: planegcs requires referenced primitives to come earlier.
  for (const e of s.entities) if (e.kind === 'point') out.push({ id: e.id, type: 'point', x: e.x, y: e.y, fixed: !!e.fixed || !!e.ref })
  for (const e of s.entities) {
    if (e.kind === 'line') out.push({ id: e.id, type: 'line', p1_id: e.p1, p2_id: e.p2 })
    else if (e.kind === 'circle') out.push({ id: e.id, type: 'circle', c_id: e.c, radius: e.r })
    else if (e.kind === 'arc') {
      const c = pt(s, e.c), a = pt(s, e.start), b = pt(s, e.end)
      const a0 = Math.atan2(a.y - c.y, a.x - c.x)
      let a1 = Math.atan2(b.y - c.y, b.x - c.x)
      if (a1 <= a0) a1 += 2 * Math.PI
      out.push({ id: e.id, type: 'arc', c_id: e.c, radius: Math.hypot(a.x - c.x, a.y - c.y), start_angle: a0, end_angle: a1, start_id: e.start, end_id: e.end })
      out.push({ id: `${e.id}__rules`, type: 'arc_rules', a_id: e.id })
    }
    // Splines: the solver moves their points; the curve through them is drawn afterwards.
  }
  for (const c of s.constraints) out.push(...(c.kind === 'tangent' ? endpointTangent(s, c) : null) ?? constraintPrims(c, kind))
  return out
}

const ends = (e: Entity | undefined): string[] => (e?.kind === 'line' ? [e.p1, e.p2] : e?.kind === 'arc' ? [e.start, e.end] : [])

/** Tangency where two curves share an endpoint is an angle of 0 or 180° between them at that point (as FreeCAD does):
 *  edge tangency plus the shared point is numerically degenerate at the solution and reads as redundant. */
function endpointTangent(s: Sketch, c: Constraint): Prim[] | null {
  const [e1, e2] = c.refs.map((id) => s.entities.find((e) => e.id === id))
  const shared = ends(e1).find((p) => ends(e2).includes(p))
  if (!shared || !e1 || !e2) return null
  const p = pt(s, shared)
  const dir = (e: Entity) => {
    if (e.kind === 'line') {
      const a = pt(s, e.p1), b = pt(s, e.p2)
      return [b.x - a.x, b.y - a.y]
    }
    const cc = pt(s, (e as { c: string }).c)
    return [-(p.y - cc.y), p.x - cc.x] // counter-clockwise along an arc
  }
  const [ax, ay] = dir(e1), [bx, by] = dir(e2)
  const angle = Math.abs(Math.atan2(ax * by - ay * bx, ax * bx + ay * by)) < Math.PI / 2 ? 0 : Math.PI
  return [{ id: c.id, type: 'angle_via_point', crv1_id: e1.id, crv2_id: e2.id, p_id: shared, angle, driving: c.driving !== false }]
}

function constraintPrims(c: Constraint, kind: Map<string, Entity['kind']>): Prim[] {
  const [a, b] = c.refs
  const ka = kind.get(a), kb = b !== undefined ? kind.get(b) : undefined
  const base = { id: c.id, driving: c.driving !== false }
  switch (c.kind) {
    case 'coincident': return [{ ...base, type: 'p2p_coincident', p1_id: a, p2_id: b }]
    case 'horizontal': return [{ ...base, type: 'horizontal_l', l_id: a }]
    case 'vertical': return [{ ...base, type: 'vertical_l', l_id: a }]
    case 'parallel': return [{ ...base, type: 'parallel', l1_id: a, l2_id: b }]
    case 'perpendicular': return [{ ...base, type: 'perpendicular_ll', l1_id: a, l2_id: b }]
    case 'angle': return [{ ...base, type: 'l2l_angle_ll', l1_id: a, l2_id: b, angle: c.value ?? 0 }]
    case 'fix': return [] // handled as a fixed point
    case 'concentric': return [] // expanded by the caller into a centre coincidence
    case 'distance':
      if (kb === 'line') return [{ ...base, type: 'p2l_distance', p_id: a, l_id: b, distance: c.value ?? 0 }]
      return [{ ...base, type: 'p2p_distance', p1_id: a, p2_id: b, distance: c.value ?? 0 }]
    case 'radius':
      return [ka === 'arc' ? { ...base, type: 'arc_radius', a_id: a, radius: c.value ?? 0 } : { ...base, type: 'circle_radius', c_id: a, radius: c.value ?? 0 }]
    case 'diameter':
      return [ka === 'arc' ? { ...base, type: 'arc_radius', a_id: a, radius: (c.value ?? 0) / 2 } : { ...base, type: 'circle_radius', c_id: a, radius: (c.value ?? 0) / 2 }]
    case 'point_on':
      if (kb === 'line') return [{ ...base, type: 'point_on_line_pl', p_id: a, l_id: b }]
      if (kb === 'arc') return [{ ...base, type: 'point_on_arc', p_id: a, a_id: b }]
      return [{ ...base, type: 'point_on_circle', p_id: a, c_id: b }]
    case 'equal':
      if (ka === 'line') return [{ ...base, type: 'equal_length', l1_id: a, l2_id: b }]
      if (ka === 'circle' && kb === 'circle') return [{ ...base, type: 'equal_radius_cc', c1_id: a, c2_id: b }]
      if (ka === 'arc' && kb === 'arc') return [{ ...base, type: 'equal_radius_aa', a1_id: a, a2_id: b }]
      return ka === 'circle' ? [{ ...base, type: 'equal_radius_ca', c1_id: a, a2_id: b }] : [{ ...base, type: 'equal_radius_ca', c1_id: b, a2_id: a }]
    case 'tangent': {
      if (ka === 'line' || kb === 'line') {
        const [l, k] = ka === 'line' ? [a, kb] : [b, ka]
        const o = ka === 'line' ? b : a
        return [k === 'arc' ? { ...base, type: 'tangent_la', l_id: l, a_id: o } : { ...base, type: 'tangent_lc', l_id: l, c_id: o }]
      }
      if (ka === 'arc' && kb === 'arc') return [{ ...base, type: 'tangent_aa', a1_id: a, a2_id: b }]
      if (ka === 'circle' && kb === 'circle') return [{ ...base, type: 'tangent_cc', c1_id: a, c2_id: b }]
      return ka === 'circle' ? [{ ...base, type: 'tangent_ca', c_id: a, a_id: b }] : [{ ...base, type: 'tangent_ca', c_id: b, a_id: a }]
    }
  }
  return []
}

/** Concentric needs the centre points: rewrite it as a centre coincidence before handing over. */
function expand(s: Sketch): Sketch {
  const centre = (id: string) => {
    const e = s.entities.find((x) => x.id === id)
    return e && (e.kind === 'circle' || e.kind === 'arc') ? e.c : id
  }
  const fixed = new Set(s.constraints.filter((c) => c.kind === 'fix').map((c) => c.refs[0]))
  return {
    ...s,
    entities: s.entities.map((e) => (e.kind === 'point' && fixed.has(e.id) ? { ...e, fixed: true } : e)),
    constraints: s.constraints.map((c) => (c.kind === 'concentric' ? { ...c, kind: 'coincident', refs: [centre(c.refs[0]), centre(c.refs[1])] } : c)),
  }
}

/** Where the mouse is pulling: points to positions, or a circle to a radius. Temporary constraints in planegcs give way
 *  to the real ones, so a fully constrained sketch does not move however hard it is dragged. */
export interface Drag {
  points?: { id: string; x: number; y: number }[]
  radius?: { id: string; r: number }
}

function dragPrims(s: Sketch, d?: Drag): Prim[] {
  const out: Prim[] = []
  for (const p of d?.points ?? []) {
    out.push({ id: `drag_x_${p.id}`, type: 'coordinate_x', p_id: p.id, x: p.x, temporary: true })
    out.push({ id: `drag_y_${p.id}`, type: 'coordinate_y', p_id: p.id, y: p.y, temporary: true })
  }
  if (d?.radius) {
    const e = s.entities.find((x) => x.id === d.radius!.id)
    if (e?.kind === 'circle') out.push({ id: 'drag_r', type: 'circle_radius', c_id: e.id, radius: d.radius.r, temporary: true })
  }
  return out
}

/** Solve: returns the sketch with moved geometry, and how well constrained it is. */
export async function solveSketch(s: Sketch, mod?: Module, drag?: Drag): Promise<{ sketch: Sketch; report: SolveReport }> {
  const m = mod ?? (await loadSolver())
  const g = new GcsWrapper(new m.GcsSystem())
  g.debug_mode = 0 // NoDebug: the solver otherwise prints to stdout
  try {
    const prims = [...toPrims(expand(s)), ...dragPrims(s, drag)]
    g.push_primitives_and_params(prims as unknown as Parameters<GcsWrapper['push_primitives_and_params']>[0])
    const status = g.solve()
    g.apply_solution()
    const solved = new Map(g.sketch_index.get_primitives().map((p) => [p.id, p as unknown as Prim]))
    const entities = s.entities.map((e): Entity => {
      const p = solved.get(e.id)
      if (!p) return e
      if (e.kind === 'point') return { ...e, x: p.x as number, y: p.y as number }
      if (e.kind === 'circle') return { ...e, r: p.radius as number }
      return e
    })
    const dof = (g.gcs as unknown as { dof: () => number }).dof()
    return {
      sketch: { ...s, entities },
      report: {
        status: status === 0 || status === 1 ? 'solved' : 'failed', // Success or Converged
        dof,
        conflicting: g.get_gcs_conflicting_constraints(),
        redundant: g.get_gcs_redundant_constraints(),
      },
    }
  } finally {
    g.destroy_gcs_module()
  }
}
