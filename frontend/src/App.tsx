// The app: header (design, mode tabs, update), toolbar, then the specification tree | the view | the AI panel, and the
// print checks along the bottom. The backend owns the model; this keeps a copy of its tree and last good part, refreshed
// whenever the server says something changed (server-sent events), whoever changed it — you, Claude Code or the chat.
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { api, type Activity, type AiStatus, type Checks, type Concept, type DesignInfo, type FeatureSpec, type FileState, type FontInfo, type Op, type PartState, type PlateMesh, type Printer, type Project, type SlicerInfo, type Tree } from './api'
import { AiPanel } from './ai/AiPanel'
import { ConceptView } from './ai/ConceptView'
import { FeatureDialog, ParamDialog, ReferenceDialog } from './edit/Dialogs'
import { ExportPane } from './export/ExportPane'
import type { FileAction } from './layout/FileMenu'
import { Header, type Mode } from './layout/Header'
import { ConfirmModal, DesignGrid, ExportModal, HistoryModal, NameModal, OpenModal, UnsavedModal, type ExportFormat } from './ui/Modals'
import { PartToolbar } from './layout/Toolbar'
import { PlatesPane } from './plates/PlatesPane'
import type { Tool } from './layout/tools'
import { SketchEditor } from './sketch/SketchEditor'
import { SpecTree, type NodeKey } from './tree/SpecTree'
import { Icon } from './ui/icons'
import { usePanelLayout } from './ui/usePanelLayout'
import { ViewportPane } from './viewport/ViewportPane'
import type { FacePick as ViewPick } from './viewer/Viewer'

type Dialog = { type: 'feature'; id: string } | { type: 'param'; name: string | null } | { type: 'ref'; id: string } | null
type NamePurpose = 'new' | 'save-as' | 'rename' | 'save-version'
type FileModal =
  | { type: 'name'; purpose: NamePurpose }
  | { type: 'unsaved'; then: () => void }
  | { type: 'open' } | { type: 'history' } | { type: 'revert' } | { type: 'export' }
  | { type: 'delete'; design: { slug: string; name: string } }
  | null
const SKETCH_KINDS = new Set(['extrude', 'revolve'])
const PICKS_AFTER_ADD: Record<string, 'faces' | 'edges'> = { shell: 'faces', text: 'faces', fillet: 'edges', chamfer: 'edges' }

export default function App() {
  const [project, setProject] = useState<Project | null>(null)
  const [designs, setDesigns] = useState<DesignInfo[] | null>(null)
  const [file, setFile] = useState<FileState | null>(null)
  const [fileModal, setFileModal] = useState<FileModal>(null)
  const [tree, setTree] = useState<Tree | null>(null)
  const [part, setPart] = useState<PartState | null>(null)
  const [catalog, setCatalog] = useState<FeatureSpec[]>([])
  const [fonts, setFonts] = useState<FontInfo[]>([])
  const [printers, setPrinters] = useState<Printer[]>([])
  const [slicers, setSlicers] = useState<SlicerInfo[]>([])
  const [checks, setChecks] = useState<Checks | null>(null)
  const [activity, setActivity] = useState<Activity[]>([])
  const [concepts, setConcepts] = useState<Concept[]>([])
  const [conceptOpen, setConceptOpen] = useState<string | null>(null) // shown on request
  const [conceptHidden, setConceptHidden] = useState<string | null>(null) // a waiting one the user put aside
  const [ai, setAi] = useState<AiStatus | null>(null)
  const [mode, setMode] = useState<Mode>('part')
  const leftPanel = usePanelLayout({ storageKey: 'pa.left', side: 'left', defaultWidth: 300, minWidth: 180, maxWidth: 580 })
  const rightPanel = usePanelLayout({ storageKey: 'pa.right', side: 'right', defaultWidth: 360, minWidth: 240, maxWidth: 640 })
  const [frontPanel, setFrontPanel] = useState<'left' | 'right'>('right')
  const [plateNo, setPlateNo] = useState(1) // the plate shown in the Plates tab
  const [plateMesh, setPlateMesh] = useState<PlateMesh | null>(null)
  const [selected, setSelected] = useState<NodeKey | null>(null)
  const [preview, setPreview] = useState<string | null>(null) // the colorway shown in the view
  const [dialog, setDialog] = useState<Dialog>(null)
  const [picking, setPicking] = useState<'faces' | 'edges' | null>(null)
  const [sketchId, setSketchId] = useState<string | null>(null)
  const [sketchMenu, setSketchMenu] = useState(false)
  const [face, setFace] = useState<ViewPick | null>(null)
  const [updating, setUpdating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<{ text: string; n: number } | null>(null) // a short note that something worked
  const say = useCallback((text: string) => setToast((t) => ({ text, n: (t?.n ?? 0) + 1 })), [])
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 4000)
    return () => clearTimeout(t)
  }, [toast])
  const [toolbarHost, setToolbarHost] = useState<HTMLDivElement | null>(null)
  const [treeHost, setTreeHost] = useState<HTMLDivElement | null>(null)
  const seen = useRef({ slug: '', rev: -1, built: -1, activity: -1 })

  const run = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    try {
      setError(null)
      return await fn()
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
      return undefined
    }
  }, [])

  const loadAll = useCallback(async () => {
    const fs = await api.file()
    setFile(fs)
    if (!fs.open) {
      setTree(null); setPart(null); setProject(null)
      setDesigns(await api.projects())
      seen.current = { slug: '', rev: -1, built: -1, activity: -1 }
      return
    }
    const [p, c, f, pr, sl, t, pt, ch, a, act] = await Promise.all([api.project(), api.catalog(), api.fonts(), api.printers(),
      api.slicers(), api.tree(), api.part(), api.checks(), api.aiStatus(), api.activity()])
    setProject(p); setCatalog(c); setFonts(f); setPrinters(pr); setSlicers(sl); setTree(t); setPart(pt); setChecks(ch); setAi(a); setActivity(act)
    api.concepts().then(setConcepts).catch(() => {})
    seen.current = { slug: p.slug, rev: t.rev, built: t.built_rev, activity: act.length }
  }, [])

  useEffect(() => { run(loadAll) }, [run, loadAll])

  // Server-sent events: refetch what changed.
  // When the backend restarts (a code change), the dev proxy answers the stream with an error and the browser gives
  // up on it for good; so reconnect by hand, and reload everything once it is back (anything may have changed).
  useEffect(() => {
    let es: EventSource
    let retry: ReturnType<typeof setTimeout> | undefined
    let lost = false
    const connect = () => {
      es = new EventSource('/api/events')
      es.onopen = () => { if (lost) { lost = false; run(loadAll) } }
      es.onerror = () => {
        if (es.readyState !== EventSource.CLOSED) return // the browser retries by itself
        lost = true
        retry = setTimeout(connect, 2000)
      }
      es.onmessage = (ev) => {
        const s = JSON.parse(ev.data) as { slug: string | null; rev: number; built_rev: number; activity: number }
        const k = seen.current
        if ((k.slug || s.slug) && (s.slug ?? '') !== k.slug) { run(loadAll); return }
        if (!s.slug) return
        if (s.rev !== k.rev || s.built_rev !== k.built) {
          api.tree().then(setTree).catch(() => {})
          api.file().then(setFile).catch(() => {})
        }
        if (s.built_rev !== k.built) {
          api.part().then(setPart).catch(() => {})
          api.checks().then(setChecks).catch(() => {})
        }
        if (s.activity !== k.activity) {
          api.activity().then(setActivity).catch(() => {})
          api.concepts().then(setConcepts).catch(() => {}) // a concept proposed or answered is logged as activity
        }
        seen.current = { slug: s.slug ?? '', rev: s.rev, built: s.built_rev, activity: s.activity }
      }
    }
    connect()
    return () => { clearTimeout(retry); es.close() }
  }, [run, loadAll])

  // While the chat works, follow it; when a turn finishes, ensure tree/part/checks are synced even if SSE missed an event.
  const wasChatRunning = useRef(false)
  useEffect(() => {
    const running = !!ai?.chat.running
    if (wasChatRunning.current && !running) {
      api.tree().then(setTree).catch(() => {})
      api.part().then(setPart).catch(() => {})
      api.checks().then(setChecks).catch(() => {})
      api.activity().then(setActivity).catch(() => {})
      api.concepts().then(setConcepts).catch(() => {})
      api.file().then(setFile).catch(() => {})
    }
    wasChatRunning.current = running
    if (!running) return
    const t = setInterval(() => api.aiStatus().then(setAi).catch(() => {}), 1000)
    return () => clearInterval(t)
  }, [ai?.chat.running])

  const refreshFile = () => api.file().then(setFile).catch(() => {})
  const changed = (r?: { tree: Tree }) => { if (r) { setTree(r.tree); refreshFile() } }
  // Colour, palette and variant changes: no rebuild, but the checks read the palette.
  const onOps = (ops: Op[], message: string) => {
    run(() => api.ops(ops, message)).then((r) => { changed(r); if (r) api.checks().then(setChecks).catch(() => {}) })
  }

  // The view shows each body in its filament's colour (from the tree, current without a rebuild), or a colorway's.
  const cw = preview && mode === 'export' ? tree?.colorways.find((c) => c.id === preview) : undefined
  const bodyColors = JSON.stringify((tree?.bodies ?? []).map((b) => [b.id, (cw && b.filament != null ? cw.colors[b.filament] : undefined) ?? b.color]))
  const shownPart = useMemo(() => {
    if (!part) return part
    const colors = new Map<string, string | null>(JSON.parse(bodyColors))
    return { ...part, bodies: part.bodies.map((b) => ({ ...b, color: colors.has(b.id) ? colors.get(b.id)! : b.color })) }
  }, [part, bodyColors])

  // A concept waiting for an answer opens by itself (until put aside); any concept opens from the AI panel.
  const waitingConcept = concepts.find((c) => c.status === 'pending')
  const shownConcept = concepts.find((c) => c.id === conceptOpen)
    ?? (waitingConcept && waitingConcept.id !== conceptHidden ? waitingConcept : undefined)

  // ---------- File menu ----------

  /** Run `fn` once unsaved changes are dealt with (Save / Don't save / Cancel). */
  const guard = (fn: () => void) => (file?.dirty ? setFileModal({ type: 'unsaved', then: fn }) : fn())
  const reopen = async () => {
    setSelected(null); setDialog(null); setMode('part'); setSketchId(null); setPicking(null)
    await run(loadAll)
  }
  const doSave = (message = '') => run(() => api.save(message)).then((f) => {
    if (f) { setFile(f); say(message ? `Saved version “${message}”` : `Saved “${f.name}”`) }
  })
  async function doExport(fmt: ExportFormat) {
    const r = await run(() => api.exportFile(fmt))
    if (!r) return
    const a = document.createElement('a')
    a.href = URL.createObjectURL(r.blob)
    a.download = r.name
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000)
    setFileModal(null)
    say(`Exported ${r.name} (downloaded, and kept in the design's exports folder)`)
  }

  // Import: the browser's file picker (any folder), then the file is uploaded into the design's references folder.
  // STEP files import as solids (an Import feature); STEP, STL and 3MF files can be references beside the design.
  const importInput = useRef<HTMLInputElement>(null)
  const refInput = useRef<HTMLInputElement>(null)
  const importNew = useRef(false)
  const pickImport = (newDesign: boolean) => { importNew.current = newDesign; importInput.current?.click() }
  const pickReference = (newDesign: boolean) => { importNew.current = newDesign; refInput.current?.click() }
  async function onImportFile(f: File | undefined, asReference: boolean) {
    for (const el of [importInput.current, refInput.current]) if (el) el.value = ''
    if (!f) return
    setUpdating(true)
    try {
      const r = await run(() => (asReference ? api.addReference(f, importNew.current) : api.importFile(f, importNew.current)))
      if (!r) return
      if (importNew.current) await reopen()
      else changed(r)
      const id = r.made.at(-1)
      if (asReference) {
        if (id) { setSelected(`ref:${id}`); setDialog({ type: 'ref', id }) }
        return
      }
      const t = await run(api.update)  // show it right away, whatever the update mode
      if (t) setTree(t)
      if (id) setSelected(`feature:${id}`)
    } finally { setUpdating(false) }
  }
  const importPicker = (
    <>
      <input ref={importInput} type="file" accept=".step,.stp" hidden onChange={(e) => onImportFile(e.target.files?.[0], false)} />
      <input ref={refInput} type="file" accept=".step,.stp,.stl,.3mf" hidden onChange={(e) => onImportFile(e.target.files?.[0], true)} />
    </>
  )

  function onFile(a: FileAction) {
    if (a === 'import') pickImport(false)
    else if (a === 'import-new') guard(() => pickImport(true))
    else if (a === 'reference') pickReference(false)
    else if (a === 'reference-new') guard(() => pickReference(true))
    else if (a === 'new') guard(() => setFileModal({ type: 'name', purpose: 'new' }))
    else if (a === 'open') guard(() => setFileModal({ type: 'open' }))
    else if (a === 'save') doSave()
    else if (a === 'save-version') setFileModal({ type: 'name', purpose: 'save-version' })
    else if (a === 'save-as') setFileModal({ type: 'name', purpose: 'save-as' })
    else if (a === 'rename') setFileModal({ type: 'name', purpose: 'rename' })
    else if (a === 'revert') setFileModal({ type: 'revert' })
    else if (a === 'history') setFileModal({ type: 'history' })
    else if (a === 'close') guard(() => { run(api.close).then(reopen) })
    else if (a === 'delete' && file?.slug) setFileModal({ type: 'delete', design: { slug: file.slug, name: file.name ?? file.slug } })
    else if (a === 'export') setFileModal({ type: 'export' })
    else if (a === 'reveal') run(api.reveal).then((r) => r && say(`Shown in Finder: ${r.path}`))
  }

  async function onName(purpose: NamePurpose, name: string) {
    setFileModal(null)
    if (purpose === 'new') { if (await run(() => api.createProject(name))) await reopen() }
    else if (purpose === 'save-as') { if (await run(() => api.saveAs(name))) { await reopen(); say(`Saved as “${name}”`) } }
    else if (purpose === 'rename') { const f = await run(() => api.rename(name)); if (f) { setFile(f); api.tree().then(setTree); say(`Renamed to “${f.name}”`) } }
    else await doSave(name)
  }

  const undo = () => run(api.undo).then(changed)
  const redo = () => run(api.redo).then(changed)

  const update = useCallback(() => {
    setUpdating(true)
    run(api.update).then((t) => t && setTree(t)).finally(() => setUpdating(false))
  }, [run])

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || !file?.open) return
      const k = e.key.toLowerCase()
      const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName)
      if (k === 'b') { e.preventDefault(); update() }
      else if (k === 's') { e.preventDefault(); onFile(e.shiftKey ? 'save-as' : 'save') }
      else if (k === 'o') { e.preventDefault(); onFile('open') }
      else if (k === 'z' && !typing && mode !== 'sketch') { e.preventDefault(); if (e.shiftKey) redo(); else undo() }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  })

  // Closing the browser tab with unsaved changes asks first (the recovery copy is kept either way).
  useEffect(() => {
    if (!file?.dirty) return
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault() }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [file?.dirty])

  // ---------- tools ----------

  const lastSketch = () => (selected?.startsWith('sketch:') ? selected.slice(7) : tree?.sketches.at(-1)?.id) ?? null

  // New features go into the selected body (or the selected feature's body).
  const currentBody = () => {
    if (selected?.startsWith('body:')) return selected.slice(5)
    if (selected?.startsWith('feature:')) return tree?.features.find((f) => f.id === selected.slice(8))?.body ?? 'B1'
    return tree?.bodies[0]?.id ?? 'B1'
  }

  async function onTool(t: Tool) {
    if (t.action === 'new-sketch') { setSketchMenu(!sketchMenu); return }
    if (t.action === 'new-parameter') { setDialog({ type: 'param', name: null }); return }
    if (t.action === 'import') { pickImport(false); return }
    if (t.action === 'reference') { pickReference(false); return }
    if (t.action === 'checks' || t.action === 'color') { setMode('export'); return }
    if (t.action === 'section') { setError('Use Cut X / Y / Z at the top of the view to section the part.'); return }
    if (!t.kind) return
    const body: Record<string, unknown> = { kind: t.kind, body: currentBody() }
    if (SKETCH_KINDS.has(t.kind)) {
      const s = lastSketch()
      if (!s) { setError('Draw a sketch first (Sketch tool), then extrude or revolve it.'); return }
      body.refs = { sketch: s }
    }
    const r = await run(() => api.addFeature(body))
    if (!r) return
    changed(r)
    const id = r.made[0]
    if (id) {
      setSelected(`feature:${id}`)
      setDialog({ type: 'feature', id })
      setPicking(PICKS_AFTER_ADD[t.kind] ?? null)
    }
  }

  async function newSketch(where: 'XY' | 'XZ' | 'YZ' | 'face') {
    setSketchMenu(false)
    let placement: Record<string, unknown> = { kind: 'plane', plane: where }
    if (where === 'face') {
      if (!face || face.faceId < 0) { setError('Click a flat face of the part first'); return }
      const pick = await run(() => api.pickFace(face.part, face.faceId))
      if (!pick) return
      placement = { kind: 'face', face: pick }
    }
    const r = await run(() => api.addSketch({ placement }))
    if (!r) return
    changed(r)
    const sid = r.made[0]
    if (sid) { setSketchId(sid); setSelected(`sketch:${sid}`); setMode('sketch') }
  }

  // ---------- picking in the view ----------

  async function onPick(p: ViewPick | null) {
    if (dialog?.type === 'feature' && picking && p) {
      const r = picking === 'edges'
        ? (p.edgeId != null ? await run(() => api.toggleEdge(dialog.id, p.part, p.edgeId!)) : undefined)
        : await run(() => api.toggleFace(dialog.id, p.part, p.faceId))
      changed(r)
      return
    }
    setFace(p)
    if (!p || !part) return
    const name = part.bodies.find((b) => b.id === p.part)?.mesh.face_names?.[p.faceId]
    if (name) setSelected(`feature:${name.split('.')[0]}`)
  }

  // ---------- selection and dialogs ----------

  const onEdit = (k: NodeKey) => {
    const [kind, id] = k.split(':')
    if (kind === 'feature') { setDialog({ type: 'feature', id }); setPicking(null) }
    else if (kind === 'param') setDialog({ type: 'param', name: id })
    else if (kind === 'ref') setDialog({ type: 'ref', id })
    else if (kind === 'sketch') { setSketchId(id); setMode('sketch') }
    else if (kind === 'body') setMode('export')
  }

  // The Plates tab lays the parts out only while it is open: the shown plate is fetched when the tab opens, another
  // plate is picked, or a new build or layout arrives (the backend keeps the result until one of those changes).
  const layoutKey = JSON.stringify([tree?.print_layout, part?.rev])
  useEffect(() => {
    if (mode !== 'plates' || !tree?.print_layout?.items) return
    let live = true
    api.plateMesh(plateNo).then((m) => { if (live) setPlateMesh(m) }).catch((e) => setError(e instanceof Error ? e.message : String(e)))
    return () => { live = false }
  }, [mode, plateNo, layoutKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const switchMode = (m: Mode) => {
    if (m === 'sketch' && !sketchId) {
      const s = lastSketch()
      if (!s) { setSketchMenu(true); setMode('part'); return }
      setSketchId(s)
    }
    setMode(m)
    setPicking(null)
  }

  const feature = dialog?.type === 'feature' ? tree?.features.find((f) => f.id === dialog.id) : undefined
  const spec = feature ? catalog.find((c) => c.kind === feature.kind) : undefined
  const param = dialog?.type === 'param' && dialog.name ? tree?.params.find((p) => p.name === dialog.name) ?? null : null
  const highlight = selected?.startsWith('feature:') ? selected.slice(8) : null
  const failedAt = tree?.failed ? tree.features.find((f) => f.status === 'fail') : undefined
  const context = selected ? selected.replace(':', ' ') : null

  const NAMES: Record<NamePurpose, { title: string; label: string; initial: string; ok: string }> = {
    new: { title: 'New design', label: 'Name', initial: 'Untitled', ok: 'Create' },
    'save-as': { title: 'Save as', label: 'New name', initial: `${file?.name ?? ''} copy`, ok: 'Save' },
    rename: { title: 'Rename design', label: 'Name', initial: file?.name ?? '', ok: 'Rename' },
    'save-version': { title: 'Save version', label: 'What changed (shown in Saved versions)', initial: '', ok: 'Save' },
  }
  const modals = (
    <>
      {fileModal?.type === 'name' && (
        <NameModal key={fileModal.purpose} {...NAMES[fileModal.purpose]} onClose={() => setFileModal(null)} onOk={(v) => onName(fileModal.purpose, v)} />
      )}
      {fileModal?.type === 'unsaved' && (
        <UnsavedModal name={file?.name ?? 'this design'} onClose={() => setFileModal(null)}
          onSave={() => { const then = fileModal.then; setFileModal(null); doSave().then(then) }}
          onDiscard={() => { const then = fileModal.then; setFileModal(null); run(() => api.recover(false)).then(then) }} />
      )}
      {fileModal?.type === 'open' && (
        <OpenModal current={file?.slug ?? null} onClose={() => setFileModal(null)}
          onOpen={(slug) => { setFileModal(null); run(() => api.openProject(slug)).then(reopen) }}
          onDelete={(d) => setFileModal({ type: 'delete', design: d })} />
      )}
      {fileModal?.type === 'history' && (
        <HistoryModal onClose={() => setFileModal(null)}
          onRestore={(v) => { setFileModal(null); run(() => api.restoreVersion(v.id)).then((r) => { changed(r); if (r) say(`Restored “${v.message}” (unsaved until you save)`) }) }} />
      )}
      {fileModal?.type === 'revert' && (
        <ConfirmModal title="Revert to saved" text="Throw away the changes since the last save? (Undo can bring them back.)" ok="Revert"
          onClose={() => setFileModal(null)} onOk={() => { setFileModal(null); run(api.revert).then((f) => { refreshFile(); api.tree().then(setTree); if (f) say('Reverted to the last save') }) }} />
      )}
      {fileModal?.type === 'delete' && (
        <ConfirmModal title="Move to Trash" text={`Move “${fileModal.design.name}” and its folder to the Trash?`} ok="Move to Trash"
          onClose={() => setFileModal(null)}
          onOk={() => { const d = fileModal.design; setFileModal(null); run(() => api.deleteProject(d.slug)).then(reopen) }} />
      )}
      {fileModal?.type === 'export' && tree && (
        <ExportModal why={tree.stale ? 'Update the model first' : tree.failed ? 'A feature failed: fix it first' : null}
          onUpdate={update} onExport={doExport} onClose={() => setFileModal(null)}
          onMore={() => { setFileModal(null); setMode('export') }} />
      )}
      {toast && <div key={toast.n} className="toast" role="status">{toast.text}</div>}
      {importPicker}
    </>
  )

  if (file && !file.open) {
    return (
      <div className="start">
        <header className="hdr"><div className="brand"><Icon name="part" size={20} color="var(--accent)" width={2} /><b>3D Print Assist</b></div></header>
        <main className="startbody">
          <div className="row"><h1>Designs</h1><span className="spacer" />
            <button className="btn" onClick={() => pickReference(true)} disabled={updating}
              title="A new design with a STEP, STL or 3MF file beside it, to look at and copy features from">{updating ? 'Loading…' : 'New with reference…'}</button>
            <button className="btn" onClick={() => pickImport(true)} disabled={updating} title="A new design starting from a STEP file's solid">New from STEP…</button>
            <button className="btn primary" onClick={() => setFileModal({ type: 'name', purpose: 'new' })}>New design</button></div>
          <DesignGrid list={designs} current={null} onOpen={(slug) => { run(() => api.openProject(slug)).then(reopen) }}
            onDelete={(d) => setFileModal({ type: 'delete', design: d })} />
          {error && <p className="err">{error}</p>}
        </main>
        {modals}
      </div>
    )
  }

  if (!tree || !file) return <div className="loading">{error ? `Cannot reach the app's backend: ${error}. Is ./dev.sh running?` : 'Loading…'}</div>

  const activePrinter = project?.settings.printer ?? ''
  const activeSlicer = project?.settings.slicer || slicers.find((s) => s.installed)?.id || slicers[0]?.id || 'prusaslicer'
  const onPrinter = (id: string) => {
    run(() => api.settings({ printer: id })).then(() => Promise.all([api.project().then(setProject), api.checks().then(setChecks), api.tree().then(setTree)]))
  }
  const onSlicer = (id: string) => {
    run(() => api.settings({ slicer: id })).then(() => api.project().then(setProject))
  }

  return (
    <div className={`app mode-${mode}`}>
      {modals}
      <Header tree={tree} mode={mode} onMode={switchMode} onUpdate={update} updating={updating} file={file} onFile={onFile} onUndo={undo} onRedo={redo}
        onUpdateMode={(m) => { run(() => api.settings({ update_mode: m })).then(() => api.tree().then(setTree)) }}
        printers={printers} printer={activePrinter} onPrinter={onPrinter}
        slicers={slicers} slicer={activeSlicer} onSlicer={onSlicer} />

      {mode === 'sketch' ? <div ref={setToolbarHost} className="toolbar-host" /> : mode === 'plates' ? <div className="toolbar-wrap" /> : (
        <div className="toolbar-wrap">
          <PartToolbar onTool={onTool} active={sketchMenu ? 'sketch' : null} />
          {sketchMenu && (
            <div className="menu" role="menu" aria-label="New sketch on">
              <span className="nm">New sketch on</span>
              {(['XY', 'XZ', 'YZ'] as const).map((p) => <button key={p} role="menuitem" className="btn" onClick={() => newSketch(p)}>{p} plane</button>)}
              <button role="menuitem" className="btn" disabled={!face || face.faceId < 0} onClick={() => newSketch('face')}
                title={face ? 'The face you clicked' : 'Click a flat face of the part first'}>Selected face</button>
              <button className="x" aria-label="Close" onClick={() => setSketchMenu(false)}>×</button>
            </div>
          )}
        </div>
      )}

      <main
        className={`main${leftPanel.collapsed && !leftPanel.floating ? ' collapsed-left' : ''}${rightPanel.collapsed && !rightPanel.floating ? ' collapsed-right' : ''}`}
        style={{
          '--left-w': leftPanel.floating ? '0px' : leftPanel.collapsed ? '32px' : `${leftPanel.width}px`,
          '--right-w': rightPanel.floating ? '0px' : rightPanel.collapsed ? '32px' : `${rightPanel.width}px`,
        } as CSSProperties}
      >
        <aside
          className={`side left${leftPanel.collapsed && !leftPanel.floating ? ' collapsed' : ''}${leftPanel.floating ? ' floating' : ''}`}
          style={leftPanel.floating ? { left: leftPanel.box.x, top: leftPanel.box.y, width: leftPanel.box.w, height: leftPanel.box.h, zIndex: frontPanel === 'left' ? 42 : 41 } : undefined}
          onPointerDown={() => setFrontPanel('left')}
          aria-label={mode === 'export' ? 'Print and export' : mode === 'plates' ? 'Print plates' : 'Specification tree'}
        >
          {leftPanel.collapsed && !leftPanel.floating ? (
            <button type="button" className="side-rail" onClick={() => leftPanel.setCollapsed(false)} aria-expanded={false}
              aria-label="Expand left panel" title="Expand left panel">
              <Icon name="chevR" size={14} />
              <span className="side-rail-label">{mode === 'export' ? 'Export' : mode === 'plates' ? 'Plates' : mode === 'sketch' ? 'Sketch' : 'Model'}</span>
            </button>
          ) : (
            <div className="sidehead">
              <button type="button" className={`side-grip${leftPanel.dragging ? ' dragging' : ''}`}
                aria-label="Move left panel. Drag to float or use arrow keys when floating."
                title="Drag to float / move panel" {...leftPanel.moveHandleProps}>
                <Icon name="grip" size={14} />
              </button>
              <h2>{mode === 'export' ? 'Export' : mode === 'plates' ? 'Plates' : mode === 'sketch' ? 'Sketch' : 'Model'}</h2>
              <span className="spacer" />
              {leftPanel.floating ? (
                <button type="button" className="side-toggle" onClick={() => leftPanel.setFloating(false)}
                  aria-label="Dock left panel" title="Dock left panel">
                  <Icon name="dock" size={14} />
                </button>
              ) : (
                <button type="button" className="side-toggle" onClick={() => leftPanel.setCollapsed(true)} aria-expanded={true}
                  aria-label="Collapse left panel" title="Collapse left panel">
                  <Icon name="chevL" size={14} />
                </button>
              )}
            </div>
          )}
          <div className="side-body" hidden={leftPanel.collapsed && !leftPanel.floating}>
            {mode === 'part' && <SpecTree tree={tree} selected={selected} onSelect={setSelected} onEdit={onEdit}
              onToggleRef={(id, visible) => onOps([{ op: 'set_reference', id, visible }], `${visible ? 'Show' : 'Hide'} ${id}`)}
              onRollback={(i) => { run(() => api.rollback(i)).then(changed) }} />}
            {mode === 'sketch' && <div ref={setTreeHost} />}
            {mode === 'plates' && <PlatesPane tree={tree} plate={plateNo} onPlate={setPlateNo} slicers={slicers} slicer={activeSlicer} onError={setError} />}
            {mode === 'export' && <ExportPane tree={tree} checks={checks} printers={printers} printer={activePrinter}
              onPrinter={onPrinter} slicers={slicers} slicer={activeSlicer} onSlicer={onSlicer}
              onAddPrinter={async (p) => {
                const created = await run(() => api.createPrinter(p))
                if (!created) return
                await api.settings({ printer: created.id })
                await Promise.all([api.printers().then(setPrinters), api.project().then(setProject), api.checks().then(setChecks), api.tree().then(setTree)])
              }}
              onOps={onOps} preview={preview} onPreview={setPreview} onError={setError} />}
            {mode === 'part' && (
              <div className="legend">
                <span><Icon name="bolt" size={12} color="var(--stale)" fill="var(--stale)" /> needs update</span>
                <span><Icon name="ok" size={12} color="var(--ok)" width={2.4} /> up to date</span>
                <span><Icon name="warn" size={12} color="var(--warn)" width={2.2} /> check fails</span>
                <span><Icon name="fail" size={12} color="var(--bad)" width={2.4} /> failed</span>
              </div>
            )}
          </div>
          {!leftPanel.collapsed && !leftPanel.floating && (
            <div className="side-resizer left" role="separator" aria-orientation="vertical"
              aria-label="Resize left panel. Use left and right arrow keys or double-click to reset."
              title="Drag to resize (double-click to reset)" tabIndex={0} {...leftPanel.dockResizeProps} />
          )}
          {leftPanel.floating && (
            <div className="float-resizer" role="separator" aria-label="Resize floating left panel"
              title="Drag to resize" {...leftPanel.floatResizeProps} />
          )}
        </aside>

        {mode === 'sketch' && sketchId ? (
          <section className="center sketch-center">
            <SketchEditor key={sketchId} sid={sketchId} tree={tree} params={tree.params} toolbarHost={toolbarHost} treeHost={treeHost}
              onChanged={setTree} onExit={() => { setMode('part'); setSketchId(null) }} />
          </section>
        ) : (
          <ViewportPane part={shownPart} references={tree.references} stale={tree.pending_count} failed={failedAt ? `${failedAt.id}: ${failedAt.error}` : null}
            highlight={highlight} picking={dialog?.type === 'feature' && mode !== 'plates' ? picking : null} onPick={onPick} onUpdate={update}
            plateMode={mode === 'plates'} plate={tree.print_layout?.items && plateMesh?.plate === plateNo ? plateMesh : null}
            edgeMarks={feature?.edges.length ? feature.edges.map((e) => e.points) : null}>
            {file.recovery && (
              <div className="recovery" role="alert">
                <span>Unsaved changes from last time were kept.</span>
                <button className="btn primary sm" onClick={() => { run(() => api.recover(true)).then(() => { refreshFile(); api.tree().then(setTree) }) }}>Restore them</button>
                <button className="btn sm" onClick={() => { run(() => api.recover(false)).then(refreshFile) }}>Discard</button>
              </div>
            )}
            {shownConcept && (
              <ConceptView key={shownConcept.id} concept={shownConcept}
                onAnswer={(decision, note) => {
                  run(() => api.answerConcept(shownConcept.id, decision, note)).then((c) => {
                    if (c) { setConceptOpen(null); api.concepts().then(setConcepts) }
                  })
                }}
                onBuild={(note) => {
                  const extra = note.trim() ? ` Additional instructions: ${note.trim()}` : ''
                  const msg = `Build concept ${shownConcept.id} ("${shownConcept.title}"). Call look_at_concept("${shownConcept.id}") to read its full specification and build it now in 3D Print Assist.${extra}`
                  setConceptOpen(null)
                  if (rightPanel.collapsed) rightPanel.setCollapsed(false)
                  run(async () => {
                    if (ai?.mode !== 'api') await api.settings({ ai_mode: 'api' })
                    await api.chat(msg)
                  }).then(() => api.aiStatus().then(setAi))
                }}
                onClose={() => { if (shownConcept.status === 'pending') setConceptHidden(shownConcept.id); setConceptOpen(null) }} />
            )}
            {feature && spec && (
              <FeatureDialog feature={feature} spec={spec} tree={tree} fonts={fonts} picking={picking} onPick={setPicking}
                onApply={(patch) => { run(() => api.editFeature(feature.id, patch)).then(changed) }}
                onDelete={() => { run(() => api.deleteFeature(feature.id)).then((r) => { changed(r); if (r) { setDialog(null); setSelected(null) } }) }}
                onEditSketch={(sid) => { setSketchId(sid); setMode('sketch') }}
                onClose={() => { setDialog(null); setPicking(null) }} />
            )}
            {dialog?.type === 'ref' && tree.references.some((r) => r.id === dialog.id) && (
              <ReferenceDialog reference={tree.references.find((r) => r.id === dialog.id)!}
                onSet={(patch) => onOps([{ op: 'set_reference', id: dialog.id, ...patch }], `Change ${dialog.id}`)}
                onDelete={() => { onOps([{ op: 'delete_reference', id: dialog.id }], `Remove ${dialog.id}`); setDialog(null); setSelected(null) }}
                onClose={() => setDialog(null)} />
            )}
            {dialog?.type === 'param' && (
              <ParamDialog key={dialog.name ?? 'new'} param={param} tree={tree}
                onSave={(p) => { run(() => api.setParameter(p)).then((r) => { changed(r); if (r && !param) setDialog({ type: 'param', name: p.name }) }) }}
                onDelete={() => { run(() => api.deleteParameter(dialog.name!)).then((r) => { changed(r); if (r) setDialog(null) }) }}
                onClose={() => setDialog(null)} />
            )}
          </ViewportPane>
        )}

        <AiPanel status={ai} activity={activity} context={context} concept={concepts[0] ?? null}
          layout={rightPanel} zIndex={frontPanel === 'right' ? 42 : 41} onFocusPanel={() => setFrontPanel('right')}
          onOpenConcept={(id) => { setConceptOpen(id); setConceptHidden(null) }}
          onSettings={(patch) => { run(() => api.settings(patch)).then(() => api.aiStatus().then(setAi)) }}
          onSend={(t) => { run(() => api.chat(t)).then(() => api.aiStatus().then(setAi)) }}
          onCancel={() => { run(api.chatCancel) }} onReset={() => { run(api.chatReset).then(() => api.aiStatus().then(setAi)) }}
          onRevert={() => { run(api.revertAi).then((r) => { if (r) setTree(r.tree) }) }} />
      </main>

      <footer className="foot" aria-label="Print checks">
        <span className="lbl">Print checks</span>
        {(checks?.part ?? []).filter((f) => f.code !== 'rules').slice(0, 5).map((f, i) => (
          <span key={i} className={`ck ${f.status}`} title={f.detail}>
            <Icon name={f.status === 'ok' ? 'ok' : f.status === 'warn' ? 'warn' : 'fail'} size={12} width={2.4} />{f.title}
          </span>
        ))}
        {checks?.stale && <span className="nm">(last update)</span>}
        <span className="spacer" />
        {error && <span className="err" role="alert">{error} <button className="x" onClick={() => setError(null)} aria-label="Dismiss">×</button></span>}
      </footer>
    </div>
  )
}
