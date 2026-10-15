// The sketch data model, shared with the backend (backend/src/print_assist/sketch/model.py mirrors it).
// Coordinates are mm in the sketch's own plane (see the backend's Placement).

export type Entity =
  | { id: string; kind: 'point'; x: number; y: number; fixed?: boolean; ref?: string | null } // ref: unused, kept for older designs
  | { id: string; kind: 'line'; p1: string; p2: string; construction?: boolean }
  | { id: string; kind: 'circle'; c: string; r: number; construction?: boolean }
  | { id: string; kind: 'arc'; c: string; start: string; end: string; construction?: boolean } // counter-clockwise start → end
  | { id: string; kind: 'spline'; points: string[]; construction?: boolean; tangent_start?: boolean; tangent_end?: boolean } // a curve through its points; an end can follow the curve it joins

export type ConstraintKind =
  | 'coincident' | 'horizontal' | 'vertical' | 'parallel' | 'perpendicular' | 'tangent' | 'concentric' | 'equal'
  | 'point_on' | 'distance' | 'radius' | 'diameter' | 'angle' | 'fix'

export interface Constraint {
  id: string
  kind: ConstraintKind
  refs: string[]
  value?: number | null // mm, or radians for angles
  expr?: string | null // a formula of parameters that sets the value (angles in degrees)
  driving?: boolean // false: a reference dimension, measured not imposed
}

export interface Sketch {
  id: string
  name: string
  entities: Entity[]
  constraints: Constraint[]
}

export interface SolveReport {
  status: 'solved' | 'failed'
  dof: number
  conflicting: string[] // constraint ids
  redundant: string[]
}
