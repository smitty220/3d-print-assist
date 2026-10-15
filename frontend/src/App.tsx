import { useEffect, useMemo, useRef, useState } from 'react'
import { api, isInterface, type AiStatus, type DesignState, type Fastener, type FaceMesh, type FeaturePatch, type FeatureSpec, type FromFaceKind, type IntentObject, type InterpretJob, type FemBearing, type FemField, type FemJob, type FemMesh, type MeshSetup, type MeshSetupState, type Finding, type IntentPatch, type PartRole, type Project, type Rules, type ShapeMesh, type Machining, type TemplateInfo, type TopoPreview, type TopoStatus, type TopoSummary, type ValidationReport } from './api'
import { AiPanel } from './AiPanel'
import { FeatureProps, LockButton } from './FeatureProps'
import { IntentProps, KIND_LABEL } from './IntentProps'
import { RulesProps } from './RulesProps'
import { SectionPanel } from './SectionPanel'
import { SketchEditor } from './SketchEditor'
import { StressPanel, BANDS, peakLabel, type ControlRef, type FemQuantity, type FemTab, type PickKind } from './StressPanel'
import { FemStatus } from './FemStatus'
import { DEFAULT_MACHINING, TopoPanel, type TopoView } from './TopoPanel'
import { findingKey, reportClass, ValidationPanel } from './ValidationPanel'
import { area, dir, len, point, type Units } from './units'
import { BASE_NORMALS, normalThroughAxis, resolveSection, type SectionBase, type SectionSpec, type Vec3 } from './viewer/sectionPlane'
import { Viewer, type FacePick, type MouseScheme, type ViewName } from './viewer/Viewer'

const STYLE = {
  mating: { color: 0x9aa4ae, opacity: 1, cap: { fill: 0xc9cfd5, line: 0x323a42 } },
  keepout: { color: 0xd8433a, opacity: 0.45, cap: { fill: 0xf3c0bb, line: 0xb02a22 } },
}

const DESIGN = '__part__'
const DESIGN_STYLE = { color: 0xb7bfc8, opacity: 1, cap: { fill: 0xe3c79a, line: 0x6b4d1e } }

const TOPO = '__topo__'
const TOPO_STYLE = { color: 0x5b93d6, opacity: 1, pickable: false, cap: { fill: 0xbfd6f2, line: 0x1f4f8a } }

const DEFAULT_SECTION: SectionSpec = { base: 'ZX', baseNormal: BASE_NORMALS.ZX, origin: [0, 0, 0], tiltA: 0, tiltB: 0, offset: 0, flip: false }

const SWATCH: Record<IntentObject['kind'], string> = {
  planar_mount: 'var(--iface)',
  bolt_pattern: 'var(--iface)',
  bore: 'var(--iface)',
  keep_out: 'var(--ko)',
  design_space: 'var(--muted)',
  load_case: '#8a4fd6',
}

type Selection = { kind: 'face'; pick: FacePick } | { kind: 'intent'; id: string } | { kind: 'feature'; id: string } | { kind: 'rules' } | { kind: 'validation'; finding: Finding | null } | { kind: 'ai' } | { kind: 'topo' } | { kind: 'sketch'; id: string } | { kind: 'fem' } | null

export default function App() {
  const vpRef = useRef<HTMLDivElement>(null)
  const viewer = useRef<Viewer | null>(null)
  const [project, setProject] = useState<Project | null>(null)
  const [meshes, setMeshes] = useState<Record<string, ShapeMesh>>({})
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [catalog, setCatalog] = useState<Fastener[]>([])
  const [materials, setMaterials] = useState<Record<string, number>>({})
  const [design, setDesign] = useState<DesignState | null>(null)
  const [featureSpecs, setFeatureSpecs] = useState<FeatureSpec[]>([])
  const [templates, setTemplates] = useState<TemplateInfo[]>([])
  const [partVisible, setPartVisible] = useState(true)
  const [busyOp, setBusy] = useState(false)
  const [report, setReport] = useState<ValidationReport | null>(null)
  const [validating, setValidating] = useState(false)
  const [ai, setAi] = useState<AiStatus | null>(null)
  const aiRev = useRef(-1)
  const [sel, setSel] = useState<Selection>(null)
  const [topo, setTopo] = useState<TopoStatus | null>(null)
  const [topoBudget, setTopoBudget] = useState(60_000)
  const [topoSetup, setTopoSetup] = useState<{ summary: TopoSummary; preview: TopoPreview } | null>(null)
  const [topoSetupErr, setTopoSetupErr] = useState<string | null>(null)
  const [topoSetupLoading, setTopoSetupLoading] = useState(false)
  const [showSetup, setShowSetup] = useState(true)
  const [topoView, setTopoView] = useState<TopoView>({ step: 0, cutoff: 0.5 })
  const [topoMass, setTopoMass] = useState<number | null>(null)
  const [interp, setInterp] = useState<InterpretJob | null>(null)
  const [fem, setFem] = useState<FemJob | null>(null) // the latest job of any kind: log, solve status
  const [femRes, setFemRes] = useState<FemJob | null>(null) // the latest analysis or sizing: the results shown
  const [femField, setFemField] = useState<FemField | null>(null)
  const [femQty, setFemQty] = useState<FemQuantity>('vm')
  const [femCase, setFemCase] = useState<string | null>(null)
  const [femScale, setFemScale] = useState(0) // deformed-shape magnification; 0 = undeformed
  const [femTab, setFemTab] = useState<FemTab>('mesh')
  const [femSetup, setFemSetup] = useState<MeshSetupState | null>(null)
  const [femMesh, setFemMesh] = useState<FemMesh | null>(null)
  const [showMesh, setShowMesh] = useState(true) // element edges, on the mesh preview and on contours
  const [femBands, setFemBands] = useState(false)
  const [edgePick, setEdgePick] = useState<string | null>(null) // the fillet / chamfer whose edges are being picked
  const edgePickRef = useRef<((pick: FacePick | null) => void) | null>(null)
  const [femPick, setFemPick] = useState<PickKind | null>(null) // picking an edge / face / point on the part for a mesh control
  const femPickRef = useRef<((pick: FacePick | null) => void) | null>(null)
  const [femActive, setFemActive] = useState<ControlRef | null>(null)
  const [femStatusOpen, setFemStatusOpen] = useState(false)
  const [showModel, setShowModel] = useState(true)
  const showModelRef = useRef(showModel)
  showModelRef.current = showModel
  const [showTrace, setShowTrace] = useState(true)
  const [machining, setMachining] = useState<Machining | null>(DEFAULT_MACHINING)
  const [units, setUnits] = useState<Units>('in')
  const [mouse, setMouse] = useState<MouseScheme>('catia')
  const [ortho, setOrtho] = useState(false)
  const [section, setSection] = useState(false)
  const [spec, setSpec] = useState<SectionSpec>(DEFAULT_SECTION)
  const [showPlane, setShowPlane] = useState(true)
  const [secRange, setSecRange] = useState(100)
  const [status, setStatus] = useState('Loading project…')
  const [error, setError] = useState<string | null>(null)

  const catalogMap = useMemo(() => new Map(catalog.map((f) => [f.part_number, f])), [catalog])
  const objects: IntentObject[] = useMemo(() => project?.intent.objects ?? [], [project])
  const roleOf = (name: string): PartRole => project?.parts[name]?.role ?? 'mating'

  // ---------- viewer lifecycle ----------

  useEffect(() => {
    const v = new Viewer(vpRef.current!)
    v.onSelect = (pick) => {
      if (edgePickRef.current) edgePickRef.current(pick) // picking edges for a blend: not a selection either
      else if (femPickRef.current) femPickRef.current(pick) // adding a mesh control: the pick is not a selection
      else setSel(pick ? { kind: 'face', pick } : null)
    }
    v.onSectionDrag = (d) => setSpec((s) => ({ ...s, offset: s.offset + d }))
    viewer.current = v
    return () => {
      v.dispose()
      viewer.current = null
    }
  }, [])

  useEffect(() => {
    if (viewer.current) viewer.current.mouseScheme = mouse
  }, [mouse])

  const selectedIntentId = sel?.kind === 'intent' ? sel.id : null
  useEffect(() => {
    viewer.current?.setIntent(objects, catalogMap, selectedIntentId)
  }, [objects, catalogMap, selectedIntentId])

  useEffect(() => {
    if (!section || !viewer.current) return
    const { normal, point } = resolveSection(spec)
    viewer.current.setSectionPlane(normal, point)
  }, [section, spec])

  // ---------- validation ----------

  // Re-check whenever the intent or the part changes. The backend caches by project state, so repeats are cheap.
  useEffect(() => {
    if (!project) return
    let stale = false
    setValidating(true)
    api.validation()
      .then((r) => { if (!stale) setReport(r) })
      .catch(() => {})
      .finally(() => { if (!stale) setValidating(false) })
    return () => { stale = true }
  }, [project, design])

  const selFinding = sel?.kind === 'validation' ? sel.finding : null
  useEffect(() => {
    viewer.current?.setFinding(selFinding)
  }, [selFinding])

  // Keep the marked finding in step with fresh results (same check, new numbers), or drop it if it went away.
  useEffect(() => {
    if (!report || sel?.kind !== 'validation' || !sel.finding) return
    const key = findingKey(sel.finding)
    const fresh = [...report.preflight, ...report.part].find((f) => findingKey(f) === key) ?? null
    if (fresh !== sel.finding) setSel({ kind: 'validation', finding: fresh })
  }, [report])

  // ---------- AI generation ----------

  const aiRunning = ai?.job?.status === 'running'
  const busy = busyOp || aiRunning

  useEffect(() => {
    api.aiStatus().then(setAi).catch(() => {})
  }, [])

  // While a run is going, poll it and pull each new candidate into the viewport.
  useEffect(() => {
    if (!aiRunning) return
    const t = window.setInterval(async () => {
      const st = await api.aiStatus().catch(() => null)
      if (!st) return
      setAi(st)
      if (st.job && st.job.design_rev !== aiRev.current) {
        aiRev.current = st.job.design_rev
        refreshDesign()
      }
      if (st.job && st.job.status !== 'running') {
        refreshDesign()
        setStatus(st.job.status === 'done' ? `AI run finished: ${st.job.summary || 'see the log'}` : `AI run ${st.job.status}${st.job.error ? `: ${st.job.error}` : ''}`)
      }
    }, 1000)
    return () => window.clearInterval(t)
  }, [aiRunning])

  async function aiGenerate(instruction: string, rounds: number) {
    const res = await run('Could not start the AI', () => api.aiGenerate(instruction, rounds))
    if (!res) return
    aiRev.current = res.job.design_rev
    setAi((a) => (a ? { ...a, job: res.job } : a))
    setStatus('AI is designing…')
  }

  async function aiCancel() {
    const res = await run('Could not stop the AI', api.aiCancel)
    if (res) setStatus('Stopping after the current step…')
  }

  async function aiRevert() {
    const res = await run('Could not undo the AI run', api.aiRevert)
    if (!res) return
    setAi((a) => (a ? { ...a, job: res.job } : a))
    refreshDesign()
    setStatus('Restored the design from before the AI run')
  }

  function openAi() {
    viewer.current?.select(null)
    setSel({ kind: 'ai' })
  }

  function openValidation() {
    viewer.current?.select(null)
    setSel({ kind: 'validation', finding: null })
  }

  function gotoRef(id: string) {
    if (design?.features.some((f) => f.id === id)) {
      viewer.current?.select(null)
      setSel({ kind: 'feature', id })
    } else if (objects.some((o) => o.id === id)) selectIntent(id)
  }

  // ---------- stress analysis ----------

  const femOpen = sel?.kind === 'fem'
  const femRunning = fem?.status === 'running'
  useEffect(() => { api.femStatus().then((r) => { setFem(r.job); setFemRes(r.results) }).catch(() => {}) }, [])
  useEffect(() => {
    if (!femRunning) return
    const t = window.setInterval(async () => {
      const r = await api.femStatus().catch(() => null)
      if (!r?.job) return
      setFem(r.job)
      setFemRes(r.results)
      if (r.job.status !== 'running') {
        const what = { size: 'Sizing', mesh: 'Meshing', plan: 'Mesh planning', analyse: 'Stress analysis' }[r.job.mode]
        setStatus(r.job.status === 'done' ? `${what} finished` : `${what} failed: ${r.job.error}`)
        if (r.job.mode === 'size') refreshDesign()
        if (r.job.mode !== 'size') loadFemMesh()
        api.femSetup().then(setFemSetup).catch(() => {})
        if (r.job.status === 'done' && r.job.mode === 'analyse') setFemTab('results')
      }
    }, 500)
    return () => window.clearInterval(t)
  }, [femRunning]) // eslint-disable-line react-hooks/exhaustive-deps
  function loadFemMesh() { api.femMesh().then(setFemMesh).catch(() => setFemMesh(null)) }
  // Setup and mesh: on opening the panel, and whenever the part changes (a control may no longer find its edge).
  useEffect(() => {
    if (!femOpen) return
    api.femSetup().then(setFemSetup).catch(() => setFemSetup(null))
    loadFemMesh()
  }, [femOpen, design]) // eslint-disable-line react-hooks/exhaustive-deps
  const femCaseId = femCase ?? femRes?.cases[0]?.id ?? null
  useEffect(() => {
    if (!femOpen || femRes?.status !== 'done' || !femCaseId || !femRes.cases.length) { setFemField(null); return }
    let stale = false
    api.femField(femCaseId, femQty).then((f) => { if (!stale) setFemField(f) }).catch(() => setFemField(null))
    return () => { stale = true }
  }, [femOpen, femRes?.status, femRes?.id, femCaseId, femQty]) // eslint-disable-line react-hooks/exhaustive-deps
  // What the viewport shows with the panel open: picking needs the part itself; the Mesh step shows the mesh (or the
  // part, before one is made) with its controls; the Solve step shows the contour. The forks hide the lugs.
  useEffect(() => {
    const v = viewer.current
    if (!v) return
    const parts = Object.keys(project?.parts ?? {})
    v.pickMode = (femOpen && femPick === 'edge') || edgePick ? 'edge' : 'face'
    const s = femSetup?.setup
    const controls = femOpen && s && (femTab === 'mesh' || femPick) ? {
      part: DESIGN,
      edges: s.edges.map((e, i) => ({ id: femSetup!.resolved.edges[i] ?? -1, n: e.n })).filter((e) => e.id >= 0),
      faces: femSetup!.resolved.faces.filter((f): f is number => f != null),
      spheres: s.spheres,
      active: femActive,
    } : null
    v.setMeshControls(controls)
    const contour = femOpen && !femPick && femTab === 'results' && femField
    const meshView = femOpen && !femPick && femTab === 'mesh' && femMesh
    if (contour) {
      const pos = femScale > 0 && femField.displacement.length === femField.positions.length
        ? femField.positions.map((x, i) => x + femScale * femField.displacement[i]) : femField.positions
      const k = femField.peak_index
      const peak = k != null && 3 * k + 2 < pos.length ? [pos[3 * k], pos[3 * k + 1], pos[3 * k + 2]] : femField.peak
      v.setContour({ positions: pos, indices: femField.indices, values: femField.values, lo: femField.min, hi: femField.max, peak,
        bands: femBands ? BANDS : 0, edges: showMesh ? femField.edges : null, peakLabel: peakLabel(femField, femCaseId, units) })
    } else v.setContour(null)
    v.setMeshPreview(meshView ? { positions: femMesh.positions, indices: femMesh.indices, edges: showMesh ? femMesh.edges : null } : null)
    if (femOpen) {
      v.setVisible(DESIGN, !contour && !meshView)
      parts.forEach((n) => v.setVisible(n, false))
    } else {
      if (!topoOpen) v.setVisible(DESIGN, partVisible)
      parts.forEach((n) => v.setVisible(n, !hidden.has(n)))
    }
  }, [femOpen, femField, femScale, femTab, femMesh, femSetup, femPick, femActive, showMesh, femBands, units, femCaseId, edgePick]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (!femOpen) setFemPick(null) }, [femOpen])
  // Picking edges for a fillet / chamfer: each click adds an edge (rebuilding the part); Esc or Done ends it.
  useEffect(() => {
    if (!edgePick) { edgePickRef.current = null; return }
    edgePickRef.current = (pick) => {
      viewer.current?.clearPick()
      if (!pick || pick.part !== DESIGN || pick.edgeId == null) return
      const fid = edgePick
      designOp(`Could not pick that edge for ${fid}`, () => api.pickFeatureEdge(fid, pick.edgeId!))
    }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setEdgePick(null) }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [edgePick]) // eslint-disable-line react-hooks/exhaustive-deps
  // Picking a mesh control on the part.
  useEffect(() => {
    if (!femPick) { femPickRef.current = null; return }
    femPickRef.current = (pick) => {
      if (!pick || pick.part !== DESIGN || !femSetup) return
      const s = femSetup.setup
      const next = { ...s }
      if (femPick === 'edge' && pick.edgeId != null) {
        const poly = design?.mesh?.edges.find((e) => e.id === pick.edgeId)
        let L = 0
        if (poly) for (let i = 3; i < poly.points.length; i += 3) L += Math.hypot(poly.points[i] - poly.points[i - 3], poly.points[i + 1] - poly.points[i - 2], poly.points[i + 2] - poly.points[i - 1])
        next.edges = [...s.edges, { edge: pick.edgeId, n: Math.max(2, Math.ceil(L / s.size)), label: `edge ${pick.edgeId}` }]
      } else if (femPick === 'face' && pick.faceId >= 0) {
        next.faces = [...s.faces, { face: pick.faceId, label: `face ${pick.faceId}` }]
      } else if (femPick === 'sphere' && pick.point) {
        next.spheres = [...s.spheres, { center: pick.point.map((x) => +x.toFixed(3)), radius: 12.7, size: +(s.size / 4).toFixed(3), label: `sphere ${s.spheres.length + 1}` }]
      } else return
      femSaveSetup(next)
      setFemPick(null)
      viewer.current?.clearPick()
    }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setFemPick(null) }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [femPick, femSetup, design]) // eslint-disable-line react-hooks/exhaustive-deps
  // The status window opens by itself when a job starts.
  useEffect(() => { if (femRunning) setFemStatusOpen(true) }, [femRunning, fem?.id])

  async function femSaveSetup(next: Omit<MeshSetup, 'source'> & { source?: string }) {
    const { source: _src, ...body } = next as MeshSetup
    const r = await run('Could not save the mesh controls', () => api.femSetupSave(body))
    if (r) { setFemSetup(r); loadFemMesh() }
  }

  async function femPropose(preset: string) {
    const r = await run('Could not propose mesh controls', () => api.femSetupPropose(preset))
    if (r) { setFemSetup(r); loadFemMesh(); setStatus(`Mesh controls from the best-practice rules (${preset})`) }
  }

  async function femPlan(instruction: string, maxMeshes: number) {
    const r = await run('Could not start Claude', () => api.femPlan(instruction, maxMeshes))
    if (!r) return
    setFem(r.job)
    setStatus('Claude is planning the mesh…')
  }

  async function femMeshStart() {
    const r = await run('Could not start meshing', api.femMeshStart)
    if (!r) return
    setFem(r.job)
    setStatus('Meshing…')
  }

  async function femSize(target: number, maxEvals: number) {
    const r = await run('Could not start sizing', () => api.femSize(target, maxEvals))
    if (!r) return
    setFem(r.job)
    setFemField(null)
    setStatus('Sizing the part…')
  }

  async function femStop() {
    const r = await run('Could not stop', api.femCancel)
    if (r?.job) setFem(r.job)
  }

  async function femRevert() {
    const r = await run('Could not revert', api.femSizeRevert)
    if (!r) return
    setFem(r.job)
    setFemRes(r.results)
    showDesign(r.design)
    setFemField(null)
    setStatus('Reverted to the design before sizing')
  }

  async function femRun(convergence: boolean, bearing: FemBearing = 'cosine', peaks = false, peaksContact = false) {
    const r = await run('Could not start the analysis', () => api.femRun('medium', convergence, bearing, peaks, peaksContact))
    if (!r) return
    setFem(r.job)
    setFemField(null)
    setStatus('Stress analysis running…')
  }

  // ---------- topology optimisation ----------

  const topoOpen = sel?.kind === 'topo'
  const topoRunning = topo?.job?.status === 'running'

  useEffect(() => {
    api.topoStatus().then((st) => {
      setTopo(st)
      setTopoBudget(st.result?.budget ?? st.default_budget)
      if (st.result) setMachining(st.result.machining && { ...DEFAULT_MACHINING, ...st.result.machining })
      setTopoView((v) => ({ ...v, step: st.result?.target_step ?? Math.max(0, (st.result?.steps.length ?? 1) - 1) }))
    }).catch(() => {})
  }, [])

  // How the design space meshes at this budget: refetched when the intent changes (the backend caches it).
  useEffect(() => {
    if (!topoOpen || !project) return
    let stale = false
    setTopoSetupLoading(true)
    const t = window.setTimeout(() => {
      api.topoSetup(topoBudget)
        .then((s) => { if (!stale) { setTopoSetup(s); setTopoSetupErr(null) } })
        .catch((e: Error) => { if (!stale) { setTopoSetup(null); setTopoSetupErr(e.message) } })
        .finally(() => { if (!stale) setTopoSetupLoading(false) })
    }, 150)
    return () => { stale = true; window.clearTimeout(t) }
  }, [topoOpen, project, topoBudget])

  useEffect(() => {
    viewer.current?.setTopoPreview(topoOpen && showSetup && topoSetup ? topoSetup.preview : null)
  }, [topoOpen, showSetup, topoSetup])

  // While the panel is open the result replaces the part in the viewport, unless the part made from it is shown on top
  // (the result then becomes a ghost behind it).
  const ghost = topoOpen && showModel && (!!design?.mesh || interp?.status === 'running')
  useEffect(() => {
    viewer.current?.setVisible(DESIGN, topoOpen ? ghost : partVisible)
    if (!topoOpen) {
      viewer.current?.removePart(TOPO)
      setTopoMass(null)
    }
  }, [topoOpen, ghost, design]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    viewer.current?.setTraceOverlay(topoOpen && showTrace && interp?.overlay ? interp.overlay : null)
  }, [topoOpen, showTrace, interp?.overlay])

  useEffect(() => {
    if (topoOpen) api.topoInterpretStatus().then((r) => setInterp(r.job)).catch(() => {})
  }, [topoOpen])

  // Follow a "make parametric model" run: each feature lands in the design as it is built.
  const interpRunning = interp?.status === 'running'
  useEffect(() => {
    if (!interpRunning) return
    let rev = -1
    const t = window.setInterval(async () => {
      const r = await api.topoInterpretStatus().catch(() => null)
      const j = r?.job
      if (!j) return
      setInterp(j)
      if (j.revision !== rev) {
        rev = j.revision
        if (j.status === 'running') {
          // The part so far, straight from the build in progress.
          api.topoInterpretPreview().then((p) => {
            viewer.current?.addPart({ ...p.mesh, name: DESIGN }, DESIGN_STYLE)
            viewer.current?.setVisible(DESIGN, showModelRef.current)
          }).catch(() => {})
        }
      }
      if (j.status !== 'running') refreshDesign()
      if (j.status !== 'running') setStatus(j.status === 'done' ? 'Parametric model made from the topology result' : `Could not make the model: ${j.error}`)
    }, 1000)
    return () => window.clearInterval(t)
  }, [interpRunning]) // eslint-disable-line react-hooks/exhaustive-deps

  async function topoInterpret(step: number, mode: 'rules' | 'ai' = 'rules', instruction = '', maxBuilds = 5, stress = false) {
    const r = await run('Could not start making the model', () => api.topoInterpret(step, mode, instruction, maxBuilds, stress))
    if (!r) return
    setInterp(r.job)
    setShowModel(true)
    setStatus(mode === 'ai' ? 'Claude is drafting the part from the topology result…' : 'Making a parametric model from the topology result…')
  }

  async function topoInterpretCancel() {
    const r = await run('Could not stop', api.topoInterpretCancel)
    if (r?.job) setInterp(r.job)
  }

  const resultSteps = topo?.result?.steps.length ?? 0
  const resultKey = `${topo?.result?.key}:${resultSteps}:${topo?.job?.id}`
  useEffect(() => {
    if (!topoOpen) return
    const live = topoView.step === 'live'
    if (!live && !resultSteps) return
    let stale = false
    const load = () => api.topoMesh(topoView.step, topoView.cutoff).then((m) => {
      if (stale) return
      setTopoMass(m.mass)
      if (!m.positions.length) return viewer.current?.removePart(TOPO)
      viewer.current?.addPart({
        name: TOPO, units: 'mm', bbox_min: [0, 0, 0], bbox_max: [0, 0, 0], edges: [],
        faces: [{ id: 0, type: 'topo', area: 0, centroid: [0, 0, 0], positions: m.positions, indices: m.indices, normal: null, axis_origin: null, axis_dir: null, radius: null }],
      }, ghost ? { ...TOPO_STYLE, opacity: 0.18 } : TOPO_STYLE)
    }).catch(() => {})
    const t = window.setTimeout(load, 120)
    const poll = live ? window.setInterval(load, 2500) : undefined
    return () => { stale = true; window.clearTimeout(t); window.clearInterval(poll) }
  }, [topoOpen, topoView, resultKey, resultSteps, ghost])

  // Poll a run; when a new step lands, follow it unless the user is watching live.
  useEffect(() => {
    if (!topoRunning) return
    const t = window.setInterval(async () => {
      const st = await api.topoStatus().catch(() => null)
      if (!st) return
      setTopo(st)
      if (st.job && st.job.status !== 'running') {
        const r = st.result
        setTopoView((v) => ({ ...v, step: r?.target_step != null && r.target_step < r.steps.length ? r.target_step : Math.max(0, (r?.steps.length ?? 1) - 1) }))
        setStatus(st.job.status === 'done' ? `Topology optimisation finished: ${r?.steps.length ?? 0} result${r?.steps.length === 1 ? '' : 's'}` : `Topology run ${st.job.status}${st.job.error ? `: ${st.job.error}` : ''}`)
      }
    }, 1000)
    return () => window.clearInterval(t)
  }, [topoRunning])

  async function topoRun(mass: number | null, sweep: boolean) {
    const st = await run('Could not start the optimisation', () => api.topoRun(topoBudget, mass, sweep, machining))
    if (!st) return
    setTopo(st)
    setTopoView((v) => ({ ...v, step: 'live' }))
    setStatus('Topology optimisation running…')
  }

  async function topoCancel() {
    const st = await run('Could not stop the optimisation', api.topoCancel)
    if (st) setStatus('Stopping after the current iteration…')
  }

  function openTopo() {
    viewer.current?.select(null)
    setSel({ kind: 'topo' })
  }

  // ---------- loading ----------

  function showPart(mesh: ShapeMesh, role: PartRole) {
    viewer.current?.addPart(mesh, STYLE[role])
    setMeshes((m) => ({ ...m, [mesh.name]: mesh }))
  }

  useEffect(() => {
    let cancelled = false
    Promise.all([api.project(), api.fasteners(), api.materials(), api.featureCatalog(), api.templates()])
      .then(async ([proj, fasteners, mats, specs, tmpls]) => {
        const loaded = await Promise.all(Object.keys(proj.parts).map(api.partMesh))
        if (cancelled) return
        setCatalog(fasteners)
        setMaterials(mats)
        setFeatureSpecs(specs)
        setTemplates(tmpls)
        setProject(proj)
        loaded.forEach((m) => showPart(m, proj.parts[m.name].role))
        viewer.current?.fit()
        setStatus(`Loaded ${loaded.length} parts, ${proj.intent.objects.length} intent objects`)
        showDesign(await api.design())
      })
      .catch((e: Error) => !cancelled && setError(`Could not reach the backend: ${e.message}`))
    return () => {
      cancelled = true
    }
  }, [])

  async function run<T>(what: string, fn: () => Promise<T>): Promise<T | undefined> {
    setError(null)
    try {
      return await fn()
    } catch (err) {
      setError(`${what}: ${(err as Error).message}`)
      return undefined
    }
  }

  async function onImport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setStatus(`Importing ${file.name}…`)
    const res = await run('Import failed', () => api.uploadPart(file))
    if (!res) return
    const added = Object.keys(res.project.parts).find((n) => !(project && n in project.parts)) ?? Object.keys(res.project.parts).find((n) => file.name.startsWith(n))
    setProject(res.project)
    const names = added ? [added] : Object.keys(res.project.parts)
    const loaded = await Promise.all(names.map(api.partMesh))
    loaded.forEach((m) => showPart(m, res.project.parts[m.name].role))
    viewer.current?.fit()
    setStatus(res.warnings.length ? `Imported ${file.name} with warnings: ${res.warnings.join('; ')}` : `Imported ${file.name}`)
  }

  async function setRole(name: string, role: PartRole) {
    const proj = await run('Could not change role', () => api.setRole(name, role))
    if (!proj) return
    setProject(proj)
    showPart(meshes[name], role)
  }

  async function removePart(name: string) {
    const proj = await run('Could not remove part', () => api.deletePart(name))
    if (!proj) return
    viewer.current?.removePart(name)
    setMeshes(({ [name]: _gone, ...rest }) => rest)
    setProject(proj)
    if (sel?.kind === 'face' && sel.pick.part === name) setSel(null)
  }

  function toggleVisible(name: string) {
    const next = new Set(hidden)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    viewer.current?.setVisible(name, !next.has(name))
    setHidden(next)
  }

  // ---------- design ----------

  function showDesign(d: DesignState) {
    setDesign(d)
    if (d.mesh) {
      viewer.current?.addPart({ ...d.mesh, name: DESIGN }, DESIGN_STYLE)
      viewer.current?.setVisible(DESIGN, partVisible)
    } else viewer.current?.removePart(DESIGN)
  }

  async function designOp(what: string, fn: () => Promise<DesignState>, select?: (d: DesignState) => string | undefined) {
    setBusy(true)
    const d = await run(what, fn)
    setBusy(false)
    if (!d) return
    showDesign(d)
    const id = select?.(d)
    if (id) setSel({ kind: 'feature', id })
    const errs = d.features.filter((f) => f.error)
    setStatus(errs.length ? `${errs.length} feature(s) need attention: ${errs.map((f) => `${f.id} ${f.error}`).join('; ')}` : `Part rebuilt: ${d.features.length} features`)
  }

  const refreshDesign = () => api.design().then(showDesign).catch(() => {})

  const [sketchHost, setSketchHost] = useState<HTMLDivElement | null>(null)
  function openSketch(id: string) {
    viewer.current?.select(null)
    setSel({ kind: 'sketch', id })
  }

  // ---------- intent ----------

  async function createFromFace(kind: FromFaceKind) {
    if (sel?.kind !== 'face') return
    const res = await run(`Could not create ${KIND_LABEL[kind].toLowerCase()}`, () => api.fromFace(kind, sel.pick.part, sel.pick.faceId))
    if (!res) return
    setProject(res.project)
    selectIntent(res.created)
    setStatus(`Created ${res.created}`)
  }

  function selectIntent(id: string) {
    viewer.current?.select(null)
    setSel({ kind: 'intent', id })
  }

  async function patchIntent(id: string, patch: IntentPatch) {
    const proj = await run('Could not update', () => api.patchIntent(id, patch))
    if (proj) {
      setProject(proj)
      refreshDesign()
    }
  }

  async function addIntent(what: 'design space' | 'load case', fn: () => Promise<{ project: Project; created: string }>) {
    const res = await run(`Could not add ${what}`, fn)
    if (!res) return
    setProject(res.project)
    selectIntent(res.created)
    setStatus(`Created ${res.created}`)
  }

  async function saveRules(rules: Rules) {
    const proj = await run('Could not save rules', () => api.putRules(rules))
    if (proj) {
      setProject(proj)
      refreshDesign()
    }
  }

  async function deleteIntent(id: string) {
    const proj = await run('Could not delete', () => api.deleteIntent(id))
    if (!proj) return
    setProject(proj)
    setSel(null)
    refreshDesign()
  }

  // ---------- section ----------

  function centerOf(): Vec3 {
    const c = viewer.current!.bounds().center
    return [c.x, c.y, c.z]
  }

  function toggleSection() {
    const v = viewer.current
    if (!v) return
    const on = !section
    if (on) {
      setSecRange(v.bounds().radius)
      setSpec((s) => (s.base === 'face' ? s : { ...s, origin: centerOf() }))
    }
    v.setSection(on)
    setSection(on)
  }

  function setBase(base: Exclude<SectionBase, 'face'>) {
    setSpec({ ...DEFAULT_SECTION, base, baseNormal: BASE_NORMALS[base], origin: centerOf() })
  }

  const designPick = sel?.kind === 'face' && sel.pick.part === DESIGN ? sel.pick : null
  const pickedFeatureId = designPick ? design?.face_names[designPick.faceId]?.split('.')[0] : undefined
  const featureId = sel?.kind === 'feature' ? sel.id : pickedFeatureId
  const selFeature = featureId ? design?.features.find((f) => f.id === featureId) : undefined
  // A fillet / chamfer's picked edges are drawn while it is selected; picking stops when it is not.
  const markEdges = selFeature && ['fillet', 'chamfer'].includes(selFeature.kind) ? selFeature.edges ?? [] : []
  useEffect(() => {
    viewer.current?.setEdgeMarks(markEdges.length ? markEdges.map((e) => e.world ?? []) : null)
  }, [JSON.stringify(markEdges.map((e) => e.world))]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (edgePick && sel?.kind !== 'feature') setEdgePick(null) }, [sel]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (edgePick && sel?.kind === 'feature' && sel.id !== edgePick) setEdgePick(null) }, [sel]) // eslint-disable-line react-hooks/exhaustive-deps
  const pick = sel?.kind === 'face' && sel.pick.part !== DESIGN ? sel.pick : null
  const face: FaceMesh | undefined = pick ? meshes[pick.part]?.faces.find((f) => f.id === pick.faceId) : undefined
  const selObj = sel?.kind === 'intent' ? objects.find((o) => o.id === sel.id) : undefined
  const usedBy = pick ? objects.filter((o) => isInterface(o) && o.face.part === pick.part && o.face.face_id === pick.faceId) : []

  function alignToFace() {
    if (!face || !pick || !viewer.current) return
    const source = `${pick.part} F${face.id}`
    if (face.type === 'plane' && face.normal) {
      const n = face.normal
      setSpec({ ...DEFAULT_SECTION, base: 'face', baseNormal: [-n[0], -n[1], -n[2]], origin: face.centroid, source: `Face ${source}` })
    } else if (face.type === 'cylinder' && face.axis_dir && face.axis_origin) {
      const n = normalThroughAxis(face.axis_dir, viewer.current.viewDir())
      setSpec({ ...DEFAULT_SECTION, base: 'face', baseNormal: n, origin: face.axis_origin, source: `Axis of ${source}` })
    }
  }

  useEffect(() => {
    if (!design || !featureId) {
      viewer.current?.setHighlight(null)
      return
    }
    const ids = design.face_names.flatMap((n, i) => (n?.startsWith(featureId + '.') ? [i] : []))
    viewer.current?.setHighlight(DESIGN, ids)
  }, [design, featureId])

  const faceGroups = useMemo(() => {
    const names = new Set<string>()
    for (const n of design?.face_names ?? []) if (n) { names.add(n.split('.')[0]); names.add(n) }
    return [...names].sort()
  }, [design])

  const worstOf = (fid: string) => {
    const f = design?.features.find((x) => x.id === fid)
    if (f?.error) return 'no'
    const cs = design?.checks.filter((c) => c.feature === fid) ?? []
    return cs.some((c) => c.status === 'fail') ? 'no' : cs.some((c) => c.status === 'warn') ? 'wn' : 'ok'
  }
  const massTarget = project?.intent.rules.mass_target ?? null

  const views: ViewName[] = ['iso', 'front', 'top', 'right']
  const hasMount = objects.some((o) => o.kind === 'planar_mount')
  const hasDesignSpace = objects.some((o) => o.kind === 'design_space')
  const groups: { title: string; kinds: IntentObject['kind'][]; empty: string; add?: { label: string; disabled: boolean; title: string; onClick: () => void } }[] = [
    {
      title: 'Design space', kinds: ['design_space'], empty: hasMount ? 'None yet' : 'Needs a planar mount first',
      add: hasDesignSpace ? undefined : { label: '+ Add', disabled: !hasMount, title: hasMount ? 'Box around the interfaces in the mount frame' : 'Create a planar mount first', onClick: () => addIntent('design space', api.addDesignSpace) },
    },
    { title: 'Interfaces', kinds: ['planar_mount', 'bolt_pattern', 'bore'], empty: 'Pick a face on a gray part to create one' },
    { title: 'Keep-outs', kinds: ['keep_out'], empty: 'Set a part’s role to keep-out' },
    {
      title: 'Loads', kinds: ['load_case'], empty: 'None yet',
      add: { label: '+ Add', disabled: !objects.some(isInterface), title: 'Load case applied at an interface', onClick: () => addIntent('load case', () => api.addLoadCase()) },
    },
  ]

  return (
    <div className="app">
      <header className="panel bar">
        <div className="brand">
          CAD Assist <span className="badge">T1–T3 · Topology</span>
        </div>
        <div className="spacer" />
        <button className="btn ai" onClick={openTopo} aria-pressed={topoOpen} title="Topology optimisation of the design space">
          ◇ Topology{topoRunning && <span className="spin" aria-label="running" />}
        </button>
        <button className="btn ai" onClick={() => { viewer.current?.select(null); setSel({ kind: 'fem' }) }} aria-pressed={femOpen} title="Stress analysis of the part (FEA)">
          ⚡ Stress{femRunning && <span className="spin" aria-label="running" />}
        </button>
        <button className="btn ai" onClick={openAi} aria-pressed={sel?.kind === 'ai'} title="Generate or improve the part with Claude">
          ✦ AI{aiRunning && <span className="spin" aria-label="working" />}
        </button>
        {report && (
          <button className={`vpill ${reportClass(report)}`} onClick={openValidation} aria-pressed={sel?.kind === 'validation'} title="Pre-flight and part checks">
            {report.counts.fail ? `✖ ${report.counts.fail} failing` : report.counts.warn ? `⚠ ${report.counts.warn} to review` : '✔ All checks pass'}
            {validating && <span className="spin" aria-label="checking" />}
          </button>
        )}
        <label className="ctl" htmlFor="units">
          Units
          <select id="units" value={units} onChange={(e) => setUnits(e.target.value as Units)}>
            <option value="in">in</option>
            <option value="mm">mm</option>
          </select>
        </label>
        <label className="ctl" htmlFor="mouse">
          Mouse
          <select id="mouse" value={mouse} onChange={(e) => setMouse(e.target.value as MouseScheme)}>
            <option value="catia">CATIA</option>
            <option value="simple">Simple</option>
          </select>
        </label>
        <label className="btn primary" htmlFor="stepfile">
          Import STEP…
        </label>
        <input id="stepfile" type="file" accept=".stp,.step" hidden onChange={onImport} />
      </header>

      <div className="main">
        <aside className="panel side" aria-label="Project tree">
          <ul className="tree">
            <li className={`node${sel?.kind === 'validation' ? ' sel' : ''}`} onClick={openValidation}>
              <span className={`stat ${reportClass(report)}`} />
              <span className="lb">Validation</span>
              {report && <span className="nm">{report.counts.fail}✖ {report.counts.warn}⚠ {report.counts.ok}✔</span>}
            </li>
            <li className={`node${topoOpen ? ' sel' : ''}`} onClick={openTopo}>
              <span className={`stat ${topo?.result ? (topo.stale ? 'wn' : 'ok') : ''}`} />
              <span className="lb">Topology</span>
              <span className="nm">{topoRunning ? 'running…' : topo?.result ? `${topo.result.steps.length} results${topo.stale ? ' · stale' : ''}` : 'not run'}</span>
            </li>
            <li className={`node${femOpen ? ' sel' : ''}`} onClick={() => { viewer.current?.select(null); setSel({ kind: 'fem' }) }}>
              <span className={`stat ${femRes?.status === 'done' && femRes.cases.length ? (femRes.stale ? 'wn' : Math.min(...femRes.cases.map((c) => c.ms_ultimate)) < 0 ? 'no' : 'ok') : ''}`} />
              <span className="lb">Stress</span>
              <span className="nm">{femRunning ? 'running…' : femRes?.status === 'done' && femRes.cases.length ? `MS ${Math.min(...femRes.cases.map((c) => c.ms_ultimate)) >= 0 ? '+' : ''}${Math.min(...femRes.cases.map((c) => c.ms_ultimate)).toFixed(2)}${femRes.stale ? ' · stale' : ''}` : 'not run'}</span>
            </li>
          </ul>
          <h2>
            Part features
            {design && design.features.length > 0 && (
              <select aria-label="Insert feature" className="insert" value="" disabled={busy}
                onChange={(e) => {
                  const kind = e.target.value
                  if (!kind) return
                  // A new fillet or chamfer starts by picking its edges.
                  designOp('Could not insert feature', () => api.addFeature(kind), (d) => { if (['fillet', 'chamfer'].includes(kind) && d.created) setEdgePick(d.created); return d.created })
                }}>
                <option value="">+ Insert…</option>
                {featureSpecs.map((f) => <option key={f.kind} value={f.kind}>{f.label}</option>)}
              </select>
            )}
          </h2>
          {design && design.features.length > 0 ? (
            <>
              <ul className="tree">
                {design.features.map((f) => (
                  <li key={f.id} className={`node${featureId === f.id ? ' sel' : ''}${f.suppressed ? ' supp' : ''}`} onClick={() => { viewer.current?.select(null); setSel({ kind: 'feature', id: f.id }) }}>
                    <span className={`stat ${worstOf(f.id)}`} title={f.error ?? 'rule checks'} />
                    <span className="nm">{f.id}</span>
                    <span className="lb">{featureSpecs.find((s) => s.kind === f.kind)?.label ?? f.kind}</span>
                    <LockButton locked={f.locked} label={f.id} onToggle={() => designOp('Could not lock', () => api.patchFeature(f.id, { locked: !f.locked }))} />
                  </li>
                ))}
              </ul>
              {design.sketches.length > 0 && (
                <ul className="tree sketches">
                  {design.sketches.map((s) => (
                    <li key={s.id} className={`node${sel?.kind === 'sketch' && sel.id === s.id ? ' sel' : ''}`} onClick={() => openSketch(s.id)}
                      title="Edit this sketch">
                      <span className={`stat ${s.report?.conflicting.length || s.report?.redundant.length ? 'no' : 'ok'}`} />
                      <span className="nm">✎ {s.id}</span>
                      <span className="lb">{s.name}</span>
                      <span className="dof">{s.report ? (s.report.dof ? `${s.report.dof} DOF` : 'fixed') : ''}</span>
                    </li>
                  ))}
                </ul>
              )}
              <div className="partsum">
                <label className="sp-check" htmlFor="part-vis">
                  <input id="part-vis" type="checkbox" checked={partVisible} onChange={(e) => { setPartVisible(e.target.checked); viewer.current?.setVisible(DESIGN, e.target.checked) }} /> Show part
                </label>
                <span className={massTarget != null && design.mass > massTarget ? 'wn' : ''}>
                  {(design.mass / (units === 'in' ? 0.45359237 : 1)).toFixed(3)} {units === 'in' ? 'lb' : 'kg'}
                  {massTarget != null && ` / ${(massTarget / (units === 'in' ? 0.45359237 : 1)).toFixed(2)} target`}
                </span>
              </div>
            </>
          ) : (
            <div className="startpart">
              <p className="note">{design?.error ?? 'No part yet. Start from a reference bracket built from your interfaces:'}</p>
              {report && !report.preflight_ready && (
                <p className="errbox">Pre-flight: {report.preflight.filter((f) => f.status === 'fail').length} intent problem(s) block any part. <button className="linkish" onClick={openValidation}>Review</button></p>
              )}
              {templates.map((t) => (
                <button key={t.name} className="btn wide" disabled={busy} title={t.summary}
                  onClick={() => designOp(`Could not build ${t.label}`, () => api.applyTemplate(t.name), () => 'F1')}>
                  {t.label}
                </button>
              ))}
            </div>
          )}
          <h2>Intent tree</h2>
          {groups.map(({ title, kinds, empty, add }) => {
            const items = objects.filter((o) => kinds.includes(o.kind))
            return (
              <div key={title}>
                <div className="grp">
                  {title}
                  {add && <button className="addbtn" disabled={add.disabled} title={add.title} onClick={add.onClick}>{add.label}</button>}
                </div>
                <ul className="tree">
                  {items.length === 0 && <li className="empty">{empty}</li>}
                  {items.map((o) => (
                    <li key={o.id} className={`node${sel?.kind === 'intent' && sel.id === o.id ? ' sel' : ''}${o.kind === 'bolt_pattern' && o.parent ? ' child' : ''}`} onClick={() => selectIntent(o.id)}>
                      <span className="sw" style={{ background: SWATCH[o.kind] }} />
                      <span className="nm">{o.id}</span>
                      <span className="lb" title={o.label}>{o.label || KIND_LABEL[o.kind]}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )
          })}
          <ul className="tree">
            <li className={`node${sel?.kind === 'rules' ? ' sel' : ''}`} onClick={() => { viewer.current?.select(null); setSel({ kind: 'rules' }) }}>
              <span className="sw" style={{ background: 'var(--muted)' }} />
              <span className="lb">Design rules · {project?.intent.rules.material ?? ''}</span>
            </li>
          </ul>
          <h2>Parts</h2>
          <ul className="tree">
            {Object.keys(project?.parts ?? {}).map((name) => (
              <li key={name} className={`node${pick?.part === name ? ' sel' : ''}`}>
                <input type="checkbox" id={`vis-${name}`} checked={!hidden.has(name)} onChange={() => toggleVisible(name)} title="Show or hide" />
                <span className="lb" title={name}>{name}</span>
                <select aria-label={`Role of ${name}`} className="role" value={roleOf(name)} onChange={(e) => setRole(name, e.target.value as PartRole)}>
                  <option value="mating">mating</option>
                  <option value="keepout">keep-out</option>
                </select>
                <button className="x" onClick={() => removePart(name)} aria-label={`Remove ${name}`} title="Remove part">×</button>
              </li>
            ))}
          </ul>
          <p className="note pad">Re-importing a STEP file with the same name replaces the part and re-finds its interfaces on the new geometry.</p>
        </aside>

        <section className="panel vpwrap" ref={vpRef} aria-label="3D viewport">
          {sel?.kind === 'sketch' && design?.sketches.some((s) => s.id === sel.id) && (
            <SketchEditor sketch={design.sketches.find((s) => s.id === sel.id)!} others={design.sketches.filter((s) => s.id !== sel.id)}
              units={units} propsHost={sketchHost} onDesign={showDesign} onExit={() => setSel(null)}
              pocket={design.features.some((f) => f.kind === 'sketch_pocket' && f.refs.sketch === sel.id)} />
          )}
          {fem && femStatusOpen && (fem.status === 'running' || fem.progress?.stages?.length) && <FemStatus job={fem} onClose={() => setFemStatusOpen(false)} />}
          {edgePick && <div className="pick-banner">Pick edges to blend for {edgePick} · click a picked edge's × in the panel to remove it · Esc when done</div>}
          {femOpen && femPick && <div className="pick-banner">Pick {femPick === 'edge' ? 'an edge' : femPick === 'face' ? 'a face to map' : 'the sphere centre'} on the part · Esc cancels</div>}
          <div className="vtools">
            <div className="grp">
              {views.map((v) => (
                <button key={v} onClick={() => viewer.current?.setView(v)}>{v[0].toUpperCase() + v.slice(1)}</button>
              ))}
              <button onClick={() => viewer.current?.fit()}>Fit</button>
            </div>
            <div className="grp">
              <button aria-pressed={ortho} onClick={() => { viewer.current?.setOrtho(!ortho); setOrtho(!ortho) }}>Ortho</button>
              <button aria-pressed={section} onClick={toggleSection}>Section</button>
            </div>
          </div>
          {section && (
            <SectionPanel
              spec={spec}
              units={units}
              range={secRange}
              showPlane={showPlane}
              pickedFace={face}
              pickedLabel={pick && face ? `${pick.part} F${face.id}` : undefined}
              onChange={setSpec}
              onShowPlane={(show) => { viewer.current?.setSectionPlaneVisible(show); setShowPlane(show) }}
              onBase={setBase}
              onAlignToFace={alignToFace}
              onLookAt={() => viewer.current?.viewNormalToSection()}
            />
          )}
          <div className="hint">
            {mouse === 'catia' ? (
              <><b>Pan</b> middle-drag · <b>Rotate</b> middle + left drag · <b>Zoom</b> hold middle, click left, drag · <b>Select</b> left-click · <b>Clear</b> Esc</>
            ) : (
              <><b>Rotate</b> left-drag · <b>Pan</b> right-drag · <b>Zoom</b> wheel · <b>Select</b> left-click · <b>Clear</b> Esc</>
            )}
          </div>
        </section>

        <aside className="panel side" aria-label="Properties">
          <h2>Properties</h2>
          <div className="props">
            {sel?.kind === 'topo' ? (
              <TopoPanel status={topo} summary={topoSetup?.summary ?? null} setupError={topoSetupErr} setupLoading={topoSetupLoading}
                budget={topoBudget} showSetup={showSetup} view={topoView} shownMass={topoMass} massTarget={massTarget} units={units} machining={machining} onMachining={setMachining}
                onBudget={setTopoBudget} onShowSetup={setShowSetup} onView={setTopoView} onRun={topoRun} onCancel={topoCancel}
                interp={interp} hasDesign={!!design?.features.length} showPart={showModel} onShowPart={setShowModel}
                showTrace={showTrace} onShowTrace={setShowTrace} onInterpret={topoInterpret} onInterpretCancel={topoInterpretCancel}
                aiDraft={design?.ai_draft ?? null} />
            ) : sel?.kind === 'ai' ? (
              <AiPanel status={ai} report={report} hasDesign={!!design?.features.length} units={units}
                onGenerate={aiGenerate} onCancel={aiCancel} onRevert={aiRevert} />
            ) : sel?.kind === 'validation' ? (
              <ValidationPanel report={report} loading={validating} selected={selFinding ? findingKey(selFinding) : null} units={units}
                onSelect={(f) => setSel({ kind: 'validation', finding: selFinding && findingKey(selFinding) === findingKey(f) ? null : f })}
                onRef={gotoRef} />
            ) : selFeature && design ? (
              <FeatureProps
                feature={selFeature}
                spec={featureSpecs.find((s) => s.kind === selFeature.kind)!}
                checks={design.checks.filter((c) => c.feature === selFeature.id)}
                objects={objects}
                features={design.features}
                faceGroups={faceGroups}
                units={units}
                onPatch={(patch: FeaturePatch) => designOp(`Could not update ${selFeature.id}`, () => api.patchFeature(selFeature.id, patch))}
                onDelete={() => { setSel(null); designOp(`Could not delete ${selFeature.id}`, () => api.deleteFeature(selFeature.id)) }}
                onEditSketch={openSketch}
                pickingEdges={edgePick === selFeature.id}
                onPickEdges={(on) => setEdgePick(on ? selFeature.id : null)}
              />
            ) : sel?.kind === 'fem' ? (
              <StressPanel job={fem} results={femRes} field={femField} quantity={femQty} caseId={femCaseId} units={units} hasPart={!!design?.features.length}
                canSize={!!design?.ai_draft} onRun={femRun} onSize={femSize} onStop={femStop} onRevert={femRevert}
                onQuantity={setFemQty} onCase={setFemCase} scale={femScale} onScale={setFemScale}
                setup={femSetup} mesh={femMesh} tab={femTab} onTab={(t) => { setFemTab(t); setFemPick(null) }} showMesh={showMesh} onShowMesh={setShowMesh}
                bands={femBands} onBands={setFemBands} picking={femPick} onPick={setFemPick} active={femActive} onActive={setFemActive}
                onSetup={femSaveSetup} onPropose={femPropose} onPlan={femPlan} onMesh={femMeshStart} onShowStatus={() => setFemStatusOpen(true)} />
            ) : sel?.kind === 'sketch' ? (
              <div ref={setSketchHost} className="sketch-props" />
            ) : sel?.kind === 'rules' && project ? (
              <RulesProps rules={project.intent.rules} materials={materials} units={units} onChange={saveRules} />
            ) : selObj ? (
              <IntentProps obj={selObj} all={objects} catalog={catalog} units={units} onPatch={(p) => patchIntent(selObj.id, p)} onDelete={() => deleteIntent(selObj.id)} />
            ) : face && pick ? (
              <>
                <div className="ttl">
                  <b>{face.type[0].toUpperCase() + face.type.slice(1)} face</b>
                  <span>{pick.part} · F{face.id}</span>
                </div>
                <dl className="kv">
                  <dt>Area</dt><dd>{area(face.area, units)}</dd>
                  <dt>Centroid</dt><dd>{point(face.centroid, units)}</dd>
                  {face.normal && (<><dt>Normal</dt><dd>{dir(face.normal)}</dd></>)}
                  {face.radius != null && (<><dt>Diameter</dt><dd>Ø{len(face.radius * 2, units)}</dd></>)}
                  {face.axis_dir && (<><dt>Axis</dt><dd>{dir(face.axis_dir)}</dd></>)}
                  {face.axis_origin && (<><dt>Axis point</dt><dd>{point(face.axis_origin, units)}</dd></>)}
                </dl>
                {usedBy.length > 0 && (
                  <p className="note">Used by {usedBy.map((o) => (
                    <button key={o.id} className="linkish" onClick={() => selectIntent(o.id)}>{o.id}</button>
                  ))}</p>
                )}
                {roleOf(pick.part) === 'mating' ? (
                  <>
                    <div className="sec">Create from this face</div>
                    <div className="actions">
                      <button className="btn" disabled={face.type !== 'plane'} onClick={() => createFromFace('planar_mount')}>Planar mount</button>
                      <button className="btn" disabled={face.type !== 'plane' && face.type !== 'cylinder'} onClick={() => createFromFace('bolt_pattern')}>Bolt pattern</button>
                      <button className="btn" disabled={face.type !== 'cylinder'} onClick={() => createFromFace('bore')}>Bore</button>
                    </div>
                    <p className="note">Bolt pattern reads every round hole through the picked face; picking one hole finds the face it passes through.</p>
                  </>
                ) : (
                  <p className="note">This part is a keep-out volume. Change its role to mating to create interfaces from its faces.</p>
                )}
              </>
            ) : (
              <p className="note">Click a face on a gray mating part to create an interface from it, or pick an item in the intent tree.</p>
            )}
          </div>
        </aside>
      </div>

      <footer className={`panel status${error ? ' err' : ''}`} role="status">{error ?? status}</footer>
    </div>
  )
}
