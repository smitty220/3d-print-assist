import type { Sketch, SolveReport } from './sketch/model'
export type Vec3 = [number, number, number]
export type Vec2 = [number, number]
export type FaceType = 'plane' | 'cylinder' | 'cone' | 'sphere' | 'torus' | 'bspline' | string

export interface FaceMesh {
  id: number
  type: FaceType
  area: number
  centroid: Vec3
  positions: number[]
  indices: number[]
  normal: Vec3 | null
  axis_origin: Vec3 | null
  axis_dir: Vec3 | null
  radius: number | null
}

export interface EdgePolyline {
  id: number
  points: number[]
}

export interface ShapeMesh {
  name: string
  units: 'mm'
  bbox_min: Vec3
  bbox_max: Vec3
  faces: FaceMesh[]
  edges: EdgePolyline[]
}

// ---------- intent model (mirrors backend/src/cad_assist/intent/models.py) ----------

export interface Frame {
  origin: Vec3
  z: Vec3
  x: Vec3
}

export interface FaceRef {
  part: string
  face_id: number
}

interface IntentBase {
  id: string
  label: string
  strength: 'hard' | 'soft'
  source: 'picked' | 'typed' | 'ai'
}

export interface PlanarMount extends IntentBase {
  kind: 'planar_mount'
  frame: Frame
  footprint: Vec2[]
  face: FaceRef
  support: 'auto' | 'free' | 'normal'
}

export interface Hole {
  center: Vec2
  diameter: number
  depth: number | null
}

export interface BoltPattern extends IntentBase {
  kind: 'bolt_pattern'
  frame: Frame
  holes: Hole[]
  fastener: string | null
  fit: 'floating' | 'fixed'
  parent: string | null
  face: FaceRef
  support: 'auto' | 'fixed' | 'free'
}

export interface Bore extends IntentBase {
  kind: 'bore'
  frame: Frame
  diameter: number
  length: number
  face: FaceRef
  support: 'auto' | 'fixed' | 'free' | 'pin' | 'link'
}

export interface KeepOut extends IntentBase {
  kind: 'keep_out'
  part: string
  clearance: number
}

export interface DesignSpace extends IntentBase {
  kind: 'design_space'
  frame: Frame
  min: Vec3
  max: Vec3
}

/** Force in N, moment in N·mm, global axes, applied at the target interface's frame origin. */
export interface LoadCase extends IntentBase {
  kind: 'load_case'
  target: string | null
  force: Vec3
  moment: Vec3
  category: 'limit' | 'ultimate'
  factor: number
}

export type IntentObject = PlanarMount | BoltPattern | Bore | KeepOut | DesignSpace | LoadCase
export type InterfaceObject = PlanarMount | BoltPattern | Bore

export function isInterface(o: IntentObject): o is InterfaceObject {
  return o.kind === 'planar_mount' || o.kind === 'bolt_pattern' || o.kind === 'bore'
}

/** Lengths in mm, ratios dimensionless, mass in kg. */
export interface Rules {
  material: string
  min_wall: number
  min_wall_preferred: number
  min_floor: number
  bolt_edge_min: number
  bolt_edge_preferred: number
  lug_edge_full_bearing: number
  lug_edge_min: number
  lug_td_min: number
  pocket_depth_tool_max: number
  wall_height_thickness_max: number
  mass_target: number | null
  ftu: number | null // MPa; null: the material table's value
  fty: number | null
}
export type FromFaceKind = 'planar_mount' | 'bolt_pattern' | 'bore'
export type PartRole = 'mating' | 'keepout'

export interface Project {
  parts: Record<string, { role: PartRole }>
  intent: { objects: IntentObject[]; rules: Rules }
}

export interface Fastener {
  part_number: string
  description: string
  shank_d: number
  head: 'socket' | 'hex'
  head_size: number
  head_height: number
  hole_close: number
  washer: string
  washer_od: number
  nut: string
  nut_af: number
}

export type IntentPatch = Partial<
  Pick<BoltPattern, 'label' | 'strength' | 'fastener' | 'fit'> &
    Pick<KeepOut, 'clearance'> &
    Pick<DesignSpace, 'min' | 'max'> &
    Pick<LoadCase, 'target' | 'force' | 'moment' | 'category' | 'factor'> & { support: Support }
>

export type Support = 'auto' | 'fixed' | 'free' | 'normal' | 'pin' | 'link'

// ---------- design (feature tree; mirrors backend/src/cad_assist/features) ----------

export type ParamValue = number | string | boolean | number[][] | number[][][]

export interface ParamSpec {
  name: string
  label: string
  quantity: 'length' | 'ratio' | 'angle' | 'choice' | 'bool' | 'profile' | 'profiles'
  default: ParamValue
  min: number | null
  max: number | null
  choices: string[]
  help: string
}

export interface RefSpec {
  role: string
  label: string
  kinds: string[]
  target: 'intent' | 'feature' | 'sketch'
}

export interface FeatureSpec {
  kind: string
  label: string
  group: string
  summary: string
  params: ParamSpec[]
  refs: RefSpec[]
}

export interface FeatureState {
  id: string
  kind: string
  params: Record<string, ParamValue>
  driven: Record<string, number | string>
  refs: Record<string, string>
  error: string | null
  suppressed: boolean
  locked: boolean
  edges?: EdgePick[]
}

/** An edge picked for a fillet or chamfer, kept by its geometry (design frame, mm); `world` is for drawing it. */
export interface EdgePick { at: number[]; tangent: number[]; length: number; faces: string[]; points: number[][]; world?: number[][] }

export interface RuleCheck {
  feature: string
  name: string
  value: number | null
  status: 'ok' | 'warn' | 'fail'
  rule: string
}

export interface DesignState {
  features: FeatureState[]
  checks: RuleCheck[]
  mass: number
  volume: number
  mesh: ShapeMesh | null
  face_names: (string | null)[]
  error: string | null
  created?: string
  sketches: (Sketch & { report?: SolveReport | null })[]
  ai_draft?: { summary: string; questions: string[] } | null // what Claude said and asked about the current design
}

// ---------- validation (mirrors backend/src/cad_assist/validate.py) ----------

export type Status = 'ok' | 'warn' | 'fail'

export interface Finding {
  stage: 'preflight' | 'part'
  code: string
  title: string
  status: Status
  detail: string
  value: number | null
  quantity: 'length' | 'area' | 'ratio' | 'percent' | 'mass' | 'count' | null
  refs: string[]
  points: Vec3[]
}

export interface ValidationReport {
  preflight: Finding[]
  part: Finding[]
  counts: Record<Status, number>
  preflight_ready: boolean
}

// ---------- AI generation (mirrors backend/src/cad_assist/ai/generate.py) ----------

export interface AiStep {
  kind: 'note' | 'build' | 'template' | 'finish' | 'error'
  text: string
  round: number
  counts: Record<Status, number> | null
  mass: number | null
  at: number
}

export interface AiJob {
  id: string
  instruction: string
  model: string
  max_rounds: number
  status: 'running' | 'done' | 'failed' | 'cancelled'
  steps: AiStep[]
  rounds: number
  design_rev: number
  summary: string
  error: string | null
  best: { round: number; counts: Record<Status, number>; mass: number } | null
  tokens: { input: number; output: number; cache_read: number }
  started: number
  finished: number | null
  can_revert: boolean
}

export interface AiStatus {
  model: string
  key: boolean
  job: AiJob | null
}

// ---------- topology optimisation (mirrors backend/src/cad_assist/topo) ----------

export interface TopoSummary {
  grid: [number, number, number]
  h: number
  budget: number
  elements: number
  design: number
  solid: number
  void: number
  full_mass: number
  supports: { id: string; kind: 'fixed' | 'normal' | 'pin' | 'link'; along: string; nodes: number }[]
  loads: { id: string; label: string; target: string; nodes: number }[]
  notes: string[]
}

/** World-mm geometry for showing the set-up: flat xyz arrays. */
export interface TopoPreview {
  h: number
  axes: [Vec3, Vec3, Vec3]
  solid: number[]
  supports: { id: string; kind: 'fixed' | 'normal' | 'pin' | 'link'; points: number[] }[]
  loads: { id: string; points: number[]; forces: number[] }[]
}

export interface Machining {
  axis: 'auto' | 'x' | 'y' | 'z'
  depths: number
  free_end: 'none' | '+' | '-'
  tool_diameter: number
}

export interface TopoStep {
  index: number
  fraction: number
  mass: number
  compliance: number[]
  iterations: number
  converged: boolean
  seconds: number
  label: string
  solid_mass: number
  solid_compliance: number[]
}

export interface TopoJob {
  id: string
  budget: number
  status: 'running' | 'done' | 'failed' | 'cancelled'
  phase: 'setup' | 'optimising' | 'done'
  fractions: number[]
  labels: string[]
  step: number
  iteration: number
  history: { step: number; it: number; c: number; vol: number; s: number }[]
  error: string | null
  started: number
  finished: number | null
  setup: TopoSummary
}

export interface TopoResult {
  key: string
  budget: number
  grid: [number, number, number]
  h: number
  target_step: number | null
  steps: TopoStep[]
  setup: TopoSummary
  machining: Machining | null
}

export interface TopoStatus {
  job: TopoJob | null
  result: TopoResult | null
  stale: boolean
  default_budget: number
}

/** The drafted sketches' curves, as world polylines. */
export interface InterpretOverlay {
  sketches: { id: string; name: string; polylines: number[][] }[]
}

export interface InterpretCheck {
  mass: number
  compliance: number[]
  ref_mass: number
  ref_compliance: number[]
  stiffness_ratio: number
  brep_mass: number
}

/** "Make parametric model": a topology result read as machined features, logged step by step. */
export interface InterpretJob {
  id: string
  step: number
  status: 'running' | 'done' | 'failed'
  log: { kind: 'read' | 'rule' | 'sketch' | 'ai' | 'build' | 'skip' | 'check' | 'question' | 'done' | 'error'; text: string; t: number }[]
  mode: 'rules' | 'ai'
  instruction: string
  tokens: { input?: number; output?: number; cache_read?: number }
  revision: number
  overlay: InterpretOverlay | null
  check: InterpretCheck | null
  error: string | null
}

// ---------- stress analysis (mirrors backend/src/cad_assist/fem) ----------

export interface FemRegion { name: string; vm: number; p1: number; at: number[]; ms_ultimate: number }
export interface FemCase {
  id: string; label: string; category: 'limit' | 'ultimate'; factor: number
  vm_max: number; p1_max: number; at: number[]; u_max: number; ms_yield: number; ms_ultimate: number; residual: number
  pins: { id: string; role: string; force: number[] }[]
  regions: FemRegion[]
  convergence?: { preset: string; dof: number; regions: Record<string, number | null> }
  bearing?: FemBearing
  contact?: { passes?: number; penetration?: number; pins?: Record<string, { p_max: number; pin_vm: number; force: number[]; statics: number[] }> }
  calibration?: Record<string, number> // region -> contact peak / cosine peak, same mesh
  peaks?: FemPeaks
}
/** A hot spot refined until its peak stops moving (backend fem/converge.py). */
export interface FemPeakSpot {
  id: string; case: string; region: string; center: number[]; radius: number
  history: { dof: number; h: number; vm: number }[] // per mesh, in the rounds' model (times factor for the reported value)
  status: 'converging' | 'converged' | 'singular' | 'limit'
  factor: number; vm: number; change: number | null; extrapolated: number | null
  ms_ultimate?: number; ms_yield?: number
}
export interface FemPeaks {
  bearing: FemBearing; factored: boolean; tol: number; spots: FemPeakSpot[]
  governing: { id: string; region: string; vm: number; status: FemPeakSpot['status']; ms_ultimate: number; ms_yield: number } | null
}
export type FemBearing = 'cosine' | 'contact'
export interface FemJob {
  id: string; preset: 'coarse' | 'medium' | 'fine'; convergence: boolean
  status: 'running' | 'done' | 'failed'
  log: { kind: string; text: string; t: number }[]
  cases: FemCase[]
  stats: { tets?: number; dof?: number; material?: { E: number; nu: number; ftu: number; fty: number; source: string }; seconds?: Record<string, number> }
  error: string | null
  stale?: boolean
  mode: 'analyse' | 'size' | 'mesh' | 'plan'
  sizing: { target: number; before_mass: number; after_mass: number | null; trials: number; feasible: boolean; can_revert: boolean } | null
  progress: FemProgress
  mesh_stats: MeshMetrics | null
  started: number
  finished: number | null
}
/** Live solver status (backend FemJob.monitor). */
export interface FemProgress {
  stage?: 'mesh' | 'loads' | 'assemble' | 'precondition' | 'iterate' | 'contact' | 'stress' | 'done' | 'failed'
  contact?: { pass: number; active: number; changed: number; iterations: number }[]
  peaks?: { round: number; spots: FemPeakSpot[] }
  run?: string
  case?: string
  dof?: number
  rtol?: number
  levels?: number
  iteration?: number
  residuals?: number[]
  stages?: { name: string; t: number; case: string | null; run: string }[]
  solves?: { run: string; case: string; iterations: number; residual: number; rtol: number }[]
  t?: number
}
export interface MeshMetrics {
  tets: number; nodes: number; dof: number; gb: number
  quality: { min: number; mean: number; below_0_2: number; histogram: number[] }
  bores: { id: string; around: number; through: number | null; thickness: number }[]
  notes: string[]
}
export interface FemField { positions: number[]; indices: number[]; values: number[]; min: number; max: number; quantity: 'vm' | 'p1' | 'u'; peak: number[]; peak_index?: number; displacement: number[]; edges: number[] }
export interface EdgeDivision { edge: number; n: number; label: string; ref?: number[]; length?: number }
export interface MappedFace { face: number; label: string; ref?: number[]; area?: number }
export interface MeshSphere { center: number[]; radius: number; size: number; label: string }
export interface MeshSetup {
  size: number; curvature: number; min_size: number
  edges: EdgeDivision[]; faces: MappedFace[]; spheres: MeshSphere[]
  source: string; notes: string[]; practices: string
}
export interface MeshSetupState {
  setup: MeshSetup
  resolved: { edges: (number | null)[]; faces: (number | null)[] }
  default_practices: string
  mesh_key: string
}
export interface FemMesh { positions: number[]; indices: number[]; edges: number[]; metrics: MeshMetrics; stale: boolean }

export interface TopoMesh {
  positions: number[]
  indices: number[]
  mass: number
  cutoff: number
  step: number | 'live'
}

export interface TemplateInfo {
  name: string
  label: string
  summary: string
}

export interface FeaturePatch {
  params?: Record<string, ParamValue>
  refs?: Record<string, string>
  edges?: EdgePick[]
  locked?: boolean
  suppressed?: boolean
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

const send = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
})

export const api = {
  project: () => fetch('/api/project').then((r) => json<Project>(r)),
  partMesh: (name: string) => fetch(`/api/project/parts/${encodeURIComponent(name)}/mesh`).then((r) => json<ShapeMesh>(r)),
  fasteners: () => fetch('/api/catalog/fasteners').then((r) => json<Fastener[]>(r)),
  uploadPart: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return fetch('/api/project/parts', { method: 'POST', body }).then((r) => json<{ project: Project; warnings: string[] }>(r))
  },
  setRole: (name: string, role: PartRole) => fetch(`/api/project/parts/${encodeURIComponent(name)}`, send('PATCH', { role })).then((r) => json<Project>(r)),
  deletePart: (name: string) => fetch(`/api/project/parts/${encodeURIComponent(name)}`, send('DELETE')).then((r) => json<Project>(r)),
  fromFace: (kind: FromFaceKind, part: string, face_id: number) =>
    fetch('/api/project/intent/from-face', send('POST', { kind, part, face_id })).then((r) => json<{ project: Project; created: string }>(r)),
  patchIntent: (id: string, patch: IntentPatch) => fetch(`/api/project/intent/${encodeURIComponent(id)}`, send('PATCH', patch)).then((r) => json<Project>(r)),
  addDesignSpace: () => fetch('/api/project/intent/design-space', send('POST')).then((r) => json<{ project: Project; created: string }>(r)),
  addLoadCase: (target?: string) => fetch('/api/project/intent/load-case', send('POST', { target: target ?? null })).then((r) => json<{ project: Project; created: string }>(r)),
  materials: () => fetch('/api/materials').then((r) => json<Record<string, number>>(r)),
  putRules: (rules: Rules) => fetch('/api/project/rules', send('PUT', rules)).then((r) => json<Project>(r)),
  design: () => fetch('/api/project/design').then((r) => json<DesignState>(r)),
  putSketch: (sid: string, s: Sketch) => fetch(`/api/project/design/sketches/${sid}`, send('PUT', { name: s.name, entities: s.entities, constraints: s.constraints }))
    .then((r) => json<DesignState & { sketch: Sketch & { report: SolveReport } }>(r)),
  offsetSketch: (sid: string, source: string, rim: number, boss_gap: number) => fetch(`/api/project/design/sketches/${sid}/offset`, send('POST', { source, rim, boss_gap }))
    .then((r) => json<DesignState & { sketch: Sketch & { report: SolveReport } }>(r)),
  topoUnderlay: () => fetch('/api/topo/underlay').then((r) => json<{ nx: number; ny: number; nz: number; h: number; x0: number; y0: number; thickness: number[] }>(r)),
  aiStatus: () => fetch('/api/ai/status').then((r) => json<AiStatus>(r)),
  aiGenerate: (instruction: string, max_rounds: number) => fetch('/api/ai/generate', send('POST', { instruction, max_rounds })).then((r) => json<{ job: AiJob }>(r)),
  aiCancel: () => fetch('/api/ai/cancel', send('POST')).then((r) => json<{ job: AiJob | null }>(r)),
  aiRevert: () => fetch('/api/ai/revert', send('POST')).then((r) => json<{ job: AiJob | null }>(r)),
  topoSetup: (budget: number) => fetch(`/api/topo/setup?budget=${budget}`).then((r) => json<{ summary: TopoSummary; preview: TopoPreview }>(r)),
  topoStatus: () => fetch('/api/topo/status').then((r) => json<TopoStatus>(r)),
  topoRun: (budget: number, mass: number | null, sweep: boolean, machining: Machining | null) => fetch('/api/topo/run', send('POST', { budget, mass, sweep, machining })).then((r) => json<TopoStatus>(r)),
  topoCancel: () => fetch('/api/topo/cancel', send('POST')).then((r) => json<TopoStatus>(r)),
  topoInterpret: (step: number, mode: 'rules' | 'ai' = 'rules', instruction = '', max_builds = 5, stress = false) =>
    fetch('/api/topo/interpret', send('POST', { step, mode, instruction, max_builds, stress })).then((r) => json<{ job: InterpretJob }>(r)),
  topoInterpretCancel: () => fetch('/api/topo/interpret/cancel', send('POST')).then((r) => json<{ job: InterpretJob | null }>(r)),
  topoInterpretStatus: () => fetch('/api/topo/interpret').then((r) => json<{ job: InterpretJob | null }>(r)),
  topoInterpretPreview: () => fetch('/api/topo/interpret/preview').then((r) => json<{ revision: number; mesh: ShapeMesh }>(r)),
  femRun: (preset: string, convergence: boolean, bearing: FemBearing = 'cosine', peaks = false, peaks_contact = false) =>
    fetch('/api/fem/run', send('POST', { preset, convergence, bearing, peaks, peaks_contact })).then((r) => json<{ job: FemJob }>(r)),
  femStatus: () => fetch('/api/fem/status').then((r) => json<{ job: FemJob | null; results: FemJob | null }>(r)),
  femSize: (target_ms: number, max_evals: number) => fetch('/api/fem/size', send('POST', { target_ms, max_evals })).then((r) => json<{ job: FemJob }>(r)),
  femCancel: () => fetch('/api/fem/cancel', send('POST')).then((r) => json<{ job: FemJob | null }>(r)),
  femSizeRevert: () => fetch('/api/fem/size/revert', send('POST')).then((r) => json<{ job: FemJob | null; results: FemJob | null; design: DesignState }>(r)),
  femSetup: () => fetch('/api/fem/setup').then((r) => json<MeshSetupState>(r)),
  femSetupSave: (s: Omit<MeshSetup, 'source'>) => fetch('/api/fem/setup', send('PUT', s)).then((r) => json<MeshSetupState>(r)),
  femSetupPropose: (preset: string) => fetch('/api/fem/setup/propose', send('POST', { preset })).then((r) => json<MeshSetupState>(r)),
  femMeshStart: () => fetch('/api/fem/mesh', send('POST')).then((r) => json<{ job: FemJob }>(r)),
  femMesh: () => fetch('/api/fem/mesh').then((r) => json<FemMesh>(r)),
  femPlan: (instruction: string, max_meshes: number) => fetch('/api/fem/plan', send('POST', { instruction, max_meshes })).then((r) => json<{ job: FemJob }>(r)),
  femField: (caseId: string, quantity: string) => fetch(`/api/fem/field?case=${encodeURIComponent(caseId)}&quantity=${quantity}`).then((r) => json<FemField>(r)),
  topoMesh: (step: number | 'live', cutoff: number) => fetch(`/api/topo/mesh?step=${step}&cutoff=${cutoff}`).then((r) => json<TopoMesh>(r)),
  validation: () => fetch('/api/project/validation').then((r) => json<ValidationReport>(r)),
  featureCatalog: () => fetch('/api/features/catalog').then((r) => json<FeatureSpec[]>(r)),
  templates: () => fetch('/api/templates').then((r) => json<TemplateInfo[]>(r)),
  applyTemplate: (name: string) => fetch('/api/project/design/template', send('POST', { name })).then((r) => json<DesignState>(r)),
  addFeature: (kind: string, refs: Record<string, string> = {}) => fetch('/api/project/design/features', send('POST', { kind, refs })).then((r) => json<DesignState>(r)),
  patchFeature: (id: string, patch: FeaturePatch) => fetch(`/api/project/design/features/${encodeURIComponent(id)}`, send('PATCH', patch)).then((r) => json<DesignState>(r)),
  pickFeatureEdge: (id: string, edge: number) => fetch(`/api/project/design/features/${encodeURIComponent(id)}/edges`, send('POST', { edge })).then((r) => json<DesignState & { picked: boolean }>(r)),
  deleteFeature: (id: string) => fetch(`/api/project/design/features/${encodeURIComponent(id)}`, send('DELETE')).then((r) => json<DesignState>(r)),
  deleteIntent: (id: string) => fetch(`/api/project/intent/${encodeURIComponent(id)}`, send('DELETE')).then((r) => json<Project>(r)),
}
