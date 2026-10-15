import { describe, expect, it } from 'vitest'
import { solveSketch } from './solve.ts'
import type { Sketch } from './model.ts'
import { addArc, addCircle, addDimension, addLine, addPolygon, addRectangle, addSlot, nearestPoint } from './tools.ts'

const empty = (): Sketch => ({ id: 'S1', name: 'Test', entities: [], constraints: [] })

describe('sketch tools', () => {
  it('holds a near-level line horizontal and reuses a snapped point', async () => {
    let r = addLine(empty(), { at: [0, 0] }, { at: [50, 1] })
    expect(r.sketch.constraints.map((c) => c.kind)).toEqual(['horizontal'])
    const r2 = addLine(r.sketch, { id: r.end }, { at: [50.5, 30] })
    expect(r2.sketch.entities.filter((e) => e.kind === 'point')).toHaveLength(3) // the corner is shared
    expect(r2.sketch.constraints.map((c) => c.kind)).toEqual(['horizontal', 'vertical'])
    expect(nearestPoint(r2.sketch, [50.2, 0.9], 1)).toBe(r.end)
    r = addLine(empty(), { at: [0, 0] }, { at: [30, 20] })
    expect(r.sketch.constraints).toHaveLength(0)
    const { sketch } = await solveSketch(r2.sketch)
    const p = sketch.entities.filter((e) => e.kind === 'point')
    expect(p[0].kind === 'point' && p[1].kind === 'point' && Math.abs(p[0].y - p[1].y)).toBeLessThan(1e-9)
  })

  it('draws a rectangle that stays square and closes', async () => {
    const r = addRectangle(empty(), [0, 0], [40, 30])
    expect(r.lines).toHaveLength(4)
    expect(r.sketch.constraints.map((c) => c.kind).sort()).toEqual(['horizontal', 'horizontal', 'vertical', 'vertical'])
    const d = addDimension(r.sketch, [r.lines[0]])
    if (typeof d === 'string') throw new Error(d)
    expect(d.sketch.constraints.at(-1)!.value).toBeCloseTo(40)
    const { report } = await solveSketch(d.sketch)
    expect(report.status).toBe('solved')
    expect(report.dof).toBe(3) // width set: height and position still free
  })

  it('draws a regular polygon that a diameter and a fixed centre fully define', async () => {
    const r = addPolygon(empty(), [0, 0], [40, 0], 6)
    expect(r.lines).toHaveLength(6)
    const kinds = r.sketch.constraints.map((c) => c.kind)
    expect(kinds.filter((k) => k === 'point_on')).toHaveLength(6)
    expect(kinds.filter((k) => k === 'equal')).toHaveLength(5)
    const d = addDimension(r.sketch, [r.circle])
    if (typeof d === 'string') throw new Error(d)
    expect(d.sketch.constraints.at(-1)!.value).toBeCloseTo(80) // diameter
    const centre = d.sketch.entities.find((e) => e.kind === 'circle')!
    const fixed: Sketch = { ...d.sketch, entities: d.sketch.entities.map((e) => (e.kind === 'point' && centre.kind === 'circle' && e.id === centre.c ? { ...e, fixed: true } : e)) }
    const { report } = await solveSketch(fixed)
    expect(report.status).toBe('solved')
    expect(report.dof).toBe(0)
    expect(report.redundant).toEqual([])
  })

  it('draws circles, arcs and slots', async () => {
    const c = addCircle(empty(), { at: [0, 0] }, 10)
    expect(c.sketch.entities.find((e) => e.id === c.id)).toMatchObject({ kind: 'circle', r: 10 })
    const a = addArc(empty(), { at: [0, 0] }, { at: [10, 0] }, [0, 25])
    const end = a.sketch.entities.find((e) => e.kind === 'point' && e.id !== 'p1' && e.id !== 'p2')!
    expect(end.kind === 'point' && Math.hypot(end.x, end.y)).toBeCloseTo(10) // the end lands on the start's radius
    const s = addSlot(empty(), [0, 0], [40, 0], 10)
    const { report } = await solveSketch(s.sketch)
    expect(report.status).toBe('solved')
    expect(report.conflicting).toEqual([])
    expect(() => addCircle(empty(), { at: [0, 0] }, 0)).toThrow()
  })

  it('refuses a dimension that fits nothing', () => {
    const r = addCircle(empty(), { at: [0, 0] }, 5)
    expect(addDimension(r.sketch, [r.id, 'p1'])).toMatch(/Pick a line/)
  })
})
