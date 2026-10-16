// The backend API (proxied at /api). Every change to the design is an op; the endpoints here wrap the common ones.
import type { Sketch, SolveReport } from './sketch/model'

export type Vec3 = [number, number, number]

// ---------- meshes (for the viewer) ----------

export interface FaceMesh {
  id: number
  type: string
  area: number
  centroid: Vec3
  positions: number[]
  indices: number[]
  normal: Vec3 | null
  axis_origin?: Vec3 | null
  axis_dir?: Vec3 | null
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
  face_names?: (string | null)[]
}

export interface PartState {
  bodies: { id: string; name: string; color: string | null; mesh: ShapeMesh }[]
  volume: number
  mass: number
  rev: number
  built_rev: number
  stale: boolean
  failed: boolean
}

// ---------- the model tree ----------

export type Status = 'ok' | 'warn' | 'fail' | 'skipped' | 'stale'
export type ParamValue = number | string | boolean | number[][] | number[][][]

export interface ParamNode {
  name: string
  quantity: 'length' | 'angle' | 'ratio' | 'count' | 'bool' | 'text'
  expr: string
  value: number | string | null
  comment: string
  relation: boolean
  status: Status
  error: string | null
  users: string[] // node keys: "feature:F1", "sketch:S1", "param:top_d"
  inputs: string[]
}

export interface FacePick { name: string | null; at: number[]; normal: number[]; area: number }
export interface EdgePick { at: number[]; tangent: number[]; length: number; faces: string[]; points: number[][] }

export interface FeatureNode {
  id: string
  kind: string
  label: string
  index: number
  body: string
  status: Status
  message: string | null
  error: string | null
  locked: boolean
  suppressed: boolean
  params: Record<string, ParamValue>
  exprs: Record<string, string>
  refs: Record<string, string>
  edges: EdgePick[]
  faces: FacePick[]
  inputs: string[]
  driven: Record<string, number | string>
}

export interface Placement {
  kind: 'plane' | 'face'
  plane: 'XY' | 'XZ' | 'YZ'
  face: FacePick | null
  offset: number
  offset_expr: string | null
  rotation: number
  flip: boolean
}

export interface SketchNode {
  id: string
  name: string
  status: Status
  error: string | null
  placement: Placement
  dof: number | null
  used_by: string[]
  inputs: string[]
}

export interface FileState {
  open: boolean
  slug?: string
  name?: string
  dirty?: boolean
  recovery?: boolean
  undo?: string | null
  redo?: string | null
}

export interface DesignInfo { slug: string; name: string; modified: number; thumbnail: boolean; recovery: boolean }

export interface Version { id: string; at: string; message: string }

export interface Tree {
  name: string
  slug: string
  dirty: boolean
  undo: string | null
  redo: string | null
  rev: number
  built_rev: number
  mode: 'manual' | 'auto'
  pending: string[]
  pending_count: number
  stale: boolean
  failed: boolean
  exportable: boolean
  rollback: number | null
  params: ParamNode[]
  features: FeatureNode[]
  sketches: SketchNode[]
  bodies: BodyNode[]
  errors: Record<string, string>
  filaments: Filament[]
  variants: Variant[]
  colorways: Colorway[]
  references: Reference[]
}

/** What the AI understood, shown before it builds: a 2D drawing, summary, assumptions and questions. */
export interface Concept {
  id: string
  title: string
  summary: string
  svg: string
  assumptions: string[]
  questions: string[]
  actor: string
  status: 'pending' | 'approved' | 'revise'
  note: string
  created: number
  answered: number | null
}

/** A file beside the design to look at and copy features from; never built or printed. */
export interface Reference {
  id: string
  name: string
  file: string
  plate: number | null
  visible: boolean
  place: 'on_bed' | 'as_in_file'
  x: number
  y: number
  z: number
  rotate_z: number
  scale: number
  color: string
}

export interface ReferenceInfo extends Reference {
  error?: string
  triangles?: number
  plates?: string[]
  part_count?: number
  bounds?: { min: number[]; max: number[]; size: number[] }
  parts?: { name: string; triangles: number; size: number[]; volume: number | null }[]
}

export interface ReferenceMesh { id: string; color: string; parts: { name: string; positions: Float32Array; indices: Uint32Array }[] }

const decode = <T extends Float32Array | Uint32Array>(b64: string, Arr: { new (b: ArrayBuffer): T }): T => {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Arr(bytes.buffer)
}

export interface BodyNode {
  id: string
  name: string
  color: string | null // what it shows in: its filament's colour, else its own
  own_color: string | null
  filament: number | null
  status: Status
  features: string[]
  fed_by: string[] // features on other bodies that put shapes into this one (split, inlay text)
}

export interface Filament { slot: number; name: string; color: string; material: string; nozzle: number }
export interface Variant { id: string; name: string; overrides: Record<string, string> }
export interface Colorway { id: string; name: string; colors: Record<string, string>; names: Record<string, string> }

// ---------- catalog ----------

export interface ParamSpec {
  name: string
  label: string
  quantity: 'length' | 'ratio' | 'angle' | 'count' | 'choice' | 'bool' | 'text' | 'profile' | 'profiles'
  default: ParamValue
  min: number | null
  max: number | null
  choices: string[]
  help: string
}

export interface RefSpec { role: string; label: string; kinds: string[]; target: 'feature' | 'sketch' }

export interface FeatureSpec {
  kind: string
  label: string
  group: string
  summary: string
  params: ParamSpec[]
  refs: RefSpec[]
}

export interface Finding {
  stage: string
  code: string
  title: string
  status: 'ok' | 'warn' | 'fail'
  detail: string
  value: number | null
  quantity: string | null
  refs: string[]
  points: Vec3[]
}

export interface Checks { part: Finding[]; counts: Record<'ok' | 'warn' | 'fail', number>; stale: boolean }

export interface Activity { actor: string; text: string; at: number; ok: boolean }

export interface ChatEntry { role: 'user' | 'assistant' | 'tool' | 'error'; text: string; tool: string | null; ok: boolean; at: number }

export interface AiStatus { model: string; key: boolean; mode: 'mcp' | 'api'; chat: { running: boolean; log: ChatEntry[]; tokens: Record<string, number> } }

export interface FontInfo { name: string; styles: string[]; license: string; source: string }

export interface Printer { id: string; name: string; bed: Vec3; nozzles: { dia: number; slots: number }[] }

export interface Project {
  slug: string
  name: string
  settings: { update_mode: 'manual' | 'auto'; printer: string; ai_mode: 'mcp' | 'api' }
  design: { sketches: (Sketch & { report?: SolveReport | null })[] }
}

export type Op = Record<string, unknown> & { op: string }

// ---------- calls ----------

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`
    try {
      const body = await r.json()
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* not JSON */ }
    throw new Error(detail)
  }
  return r.json() as Promise<T>
}

const get = <T>(path: string) => fetch(`/api${path}`).then((r) => json<T>(r))
const send = <T>(method: string, path: string, body?: unknown) =>
  fetch(`/api${path}`, { method, headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
    .then((r) => json<T>(r))

type Changed = { made: (string | null)[]; tree: Tree }

export const api = {
  project: () => get<Project>('/project'),
  projects: () => get<DesignInfo[]>('/projects'),
  thumbnailUrl: (slug: string, v = 0) => `/api/projects/${encodeURIComponent(slug)}/thumbnail?v=${v}`,
  deleteProject: (slug: string) => send<{ trashed: string }>('DELETE', `/projects/${encodeURIComponent(slug)}`),
  file: () => get<FileState>('/file'),
  save: (message = '') => send<FileState>('POST', '/file/save', { message }),
  saveAs: (name: string) => send<FileState>('POST', '/file/save-as', { name }),
  rename: (name: string) => send<FileState>('POST', '/file/rename', { name }),
  revert: () => send<FileState>('POST', '/file/revert'),
  recover: (keep: boolean) => send<FileState>('POST', '/file/recover', { keep }),
  close: () => send<FileState>('POST', '/file/close'),
  reveal: () => send<{ path: string }>('POST', '/file/reveal'),
  undo: () => send<{ undone: string; tree: Tree }>('POST', '/undo'),
  redo: () => send<{ redone: string; tree: Tree }>('POST', '/redo'),
  restoreVersion: (id: string) => send<{ tree: Tree }>('POST', `/history/${id}/restore`),
  createProject: (name: string) => send<Project>('POST', '/projects', { name }),
  openProject: (slug: string) => send<Project>('POST', `/projects/${encodeURIComponent(slug)}/open`),
  history: () => get<Version[]>('/history'),
  printers: () => get<Printer[]>('/printers'),
  settings: (s: Partial<Project['settings']>) => send<Project['settings']>('PUT', '/settings', s),
  catalog: () => get<FeatureSpec[]>('/features/catalog'),
  fonts: () => get<FontInfo[]>('/fonts'),

  tree: () => get<Tree>('/tree'),
  part: () => get<PartState>('/part'),
  update: () => send<Tree>('POST', '/update'),
  checks: () => get<Checks>('/checks'),
  activity: (since = 0) => get<Activity[]>(`/activity?since=${since}`),

  ops: (ops: Op[], message = '') => send<Changed>('POST', '/ops', { ops, message }),
  setParameter: (p: { name: string; expr: string; quantity?: string; comment?: string }) => send<Changed>('POST', '/parameters', p),
  deleteParameter: (name: string) => send<Changed>('DELETE', `/parameters/${encodeURIComponent(name)}`),
  addFeature: (f: Record<string, unknown>) => send<Changed>('POST', '/features', f),
  editFeature: (id: string, patch: Record<string, unknown>) => send<Changed>('PATCH', `/features/${id}`, patch),
  deleteFeature: (id: string) => send<Changed>('DELETE', `/features/${id}`),
  toggleEdge: (id: string, body: string, index: number) => send<Changed>('POST', `/features/${id}/edges`, { body, index }),
  toggleFace: (id: string, body: string, index: number) => send<Changed>('POST', `/features/${id}/faces`, { body, index }),
  pickFace: (body: string, face: number) => get<FacePick>(`/pick/face?body=${body}&face=${face}`),
  sketch: (id: string) => get<Sketch & { report?: SolveReport | null }>(`/sketches/${id}`),
  addSketch: (s: Record<string, unknown>) => send<Changed>('POST', '/sketches', s),
  putSketch: (id: string, s: Partial<Sketch> & { placement?: Placement }) =>
    send<Changed & { sketch: Sketch & { report?: SolveReport | null } }>('PUT', `/sketches/${id}`, s),
  deleteSketch: (id: string) => send<Changed>('DELETE', `/sketches/${id}`),
  editBody: (id: string, patch: Record<string, unknown>) => send<Changed>('PATCH', `/bodies/${id}`, patch),
  deleteBody: (id: string) => send<Changed>('DELETE', `/bodies/${id}`),
  rollback: (index: number | null) => send<Changed>('PUT', '/rollback', { index }),
  evalExpr: (expr: string) => send<{ value?: number; error?: string }>('POST', '/expr/eval', { expr }),
  concepts: () => get<Concept[]>('/concepts'),
  answerConcept: (id: string, decision: 'approve' | 'revise', note: string) =>
    send<Concept>('POST', `/concepts/${id}/answer`, { decision, note }),
  addReference: (file: File, newDesign: boolean) => {
    const form = new FormData()
    form.append('file', file)
    return fetch(`/api/references?new_design=${newDesign}`, { method: 'POST', body: form }).then((r) => json<Changed>(r))
  },
  references: () => get<ReferenceInfo[]>('/references'),
  referenceMesh: async (id: string): Promise<ReferenceMesh> => {
    const r = await get<{ id: string; color: string; parts: { name: string; positions: string; indices: string }[] }>(`/references/${id}/mesh`)
    return { ...r, parts: r.parts.map((p) => ({ name: p.name, positions: decode(p.positions, Float32Array), indices: decode(p.indices, Uint32Array) })) }
  },
  importFile: (file: File, newDesign: boolean) => {
    const form = new FormData()
    form.append('file', file)
    return fetch(`/api/import?new_design=${newDesign}`, { method: 'POST', body: form }).then((r) => json<Changed>(r))
  },
  exportUrl: (fmt: '3mf' | 'stl' | 'step') => `/api/export/${fmt}`,
  exportFile: async (fmt: '3mf' | 'stl' | 'step') => {
    const r = await fetch(`/api/export/${fmt}`)
    if (!r.ok) await json(r)
    const name = /filename="?([^";]+)"?/.exec(r.headers.get('Content-Disposition') ?? '')?.[1] ?? `design.${fmt}`
    return { blob: await r.blob(), name, saved: r.headers.get('X-Saved-Path') ?? '' }
  },
  exportBatch: async (fmt: '3mf' | 'stl' | 'step', rows: { variant?: string | null; colorway?: string | null }[] = []) => {
    const r = await fetch('/api/export/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ fmt, rows }) })
    if (!r.ok) await json(r)
    const name = /filename="([^"]+)"/.exec(r.headers.get('Content-Disposition') ?? '')?.[1] ?? 'designs.zip'
    return { blob: await r.blob(), name, saved: r.headers.get('X-Saved-Path') ?? '' }
  },
  revertAi: () => send<{ reverted: string; tree: Tree }>('POST', '/revert-ai'),

  aiStatus: () => get<AiStatus>('/ai/status'),
  chat: (message: string) => send<AiStatus['chat']>('POST', '/ai/chat', { message }),
  chatCancel: () => send<AiStatus['chat']>('POST', '/ai/chat/cancel'),
  chatReset: () => send<AiStatus['chat']>('POST', '/ai/chat/reset'),
}
