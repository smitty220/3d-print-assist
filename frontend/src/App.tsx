// The app: header (design, mode tabs, update), toolbar, then the specification tree | the view | the AI panel, and the
// print checks along the bottom. The backend owns the model; this keeps a copy of its tree and last good part, refreshed
// whenever the server says something changed (server-sent events), whoever changed it — you, Claude Code or the chat.
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Activity, type AiStatus, type Checks, type FeatureSpec, type FontInfo, type PartState, type Printer, type Project, type Tree } from './api'
import { AiPanel } from './ai/AiPanel'
import { FeatureDialog, ParamDialog } from './edit/Dialogs'
import { ExportPane } from './export/ExportPane'
import { Header, type Mode } from './layout/Header'
import { PartToolbar } from './layout/Toolbar'
import type { Tool } from './layout/tools'
import { SketchEditor } from './sketch/SketchEditor'
import { SpecTree, type NodeKey } from './tree/SpecTree'
import { Icon } from './ui/icons'
import { ViewportPane } from './viewport/ViewportPane'
import type { FacePick as ViewPick } from './viewer/Viewer'

type Dialog = { type: 'feature'; id: string } | { type: 'param'; name: string | null } | null
const SKETCH_KINDS = new Set(['extrude', 'revolve'])
const PICKS_AFTER_ADD: Record<string, 'faces' | 'edges'> = { shell: 'faces', text: 'faces', fillet: 'edges', chamfer: 'edges' }

export default function App() {
  const [project, setProject] = useState<Project | null>(null)
  const [designs, setDesigns] = useState<{ slug: string; name: string }[]>([])
  const [tree, setTree] = useState<Tree | null>(null)
  const [part, setPart] = useState<PartState | null>(null)
  const [catalog, setCatalog] = useState<FeatureSpec[]>([])
  const [fonts, setFonts] = useState<FontInfo[]>([])
  const [printers, setPrinters] = useState<Printer[]>([])
  const [checks, setChecks] = useState<Checks | null>(null)
  const [activity, setActivity] = useState<Activity[]>([])
  const [ai, setAi] = useState<AiStatus | null>(null)
  const [mode, setMode] = useState<Mode>('part')
  const [selected, setSelected] = useState<NodeKey | null>(null)
  const [dialog, setDialog] = useState<Dialog>(null)
  const [picking, setPicking] = useState<'faces' | 'edges' | null>(null)
  const [sketchId, setSketchId] = useState<string | null>(null)
  const [sketchMenu, setSketchMenu] = useState(false)
  const [face, setFace] = useState<ViewPick | null>(null)
  const [updating, setUpdating] = useState(false)
  const [error, setError] = useState<string | null>(null)
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
    const [p, d, c, f, pr, t, pt, ch, a, act] = await Promise.all([api.project(), api.projects(), api.catalog(), api.fonts(), api.printers(),
      api.tree(), api.part(), api.checks(), api.aiStatus(), api.activity()])
    setProject(p); setDesigns(d); setCatalog(c); setFonts(f); setPrinters(pr); setTree(t); setPart(pt); setChecks(ch); setAi(a); setActivity(act)
    seen.current = { slug: p.slug, rev: t.rev, built: t.built_rev, activity: act.length }
  }, [])

  useEffect(() => { run(loadAll) }, [run, loadAll])

  // Server-sent events: refetch what changed.
  useEffect(() => {
    const es = new EventSource('/api/events')
    es.onmessage = (ev) => {
      const s = JSON.parse(ev.data) as { slug: string; rev: number; built_rev: number; activity: number }
      const k = seen.current
      if (k.slug && s.slug !== k.slug) { run(loadAll); return }
      if (s.rev !== k.rev || s.built_rev !== k.built) api.tree().then(setTree).catch(() => {})
      if (s.built_rev !== k.built) {
        api.part().then(setPart).catch(() => {})
        api.checks().then(setChecks).catch(() => {})
      }
      if (s.activity !== k.activity) api.activity().then(setActivity).catch(() => {})
      seen.current = { slug: s.slug, rev: s.rev, built: s.built_rev, activity: s.activity }
    }
    return () => es.close()
  }, [run, loadAll])

  // While the chat works, follow it.
  useEffect(() => {
    if (!ai?.chat.running) return
    const t = setInterval(() => api.aiStatus().then(setAi).catch(() => {}), 1000)
    return () => clearInterval(t)
  }, [ai?.chat.running])

  const changed = (r?: { tree: Tree }) => { if (r) setTree(r.tree) }

  const update = useCallback(() => {
    setUpdating(true)
    run(api.update).then((t) => t && setTree(t)).finally(() => setUpdating(false))
  }, [run])

  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key.toLowerCase() === 'b' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); update() } }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [update])

  // ---------- tools ----------

  const lastSketch = () => (selected?.startsWith('sketch:') ? selected.slice(7) : tree?.sketches.at(-1)?.id) ?? null

  async function onTool(t: Tool) {
    if (t.action === 'new-sketch') { setSketchMenu(!sketchMenu); return }
    if (t.action === 'new-parameter') { setDialog({ type: 'param', name: null }); return }
    if (t.action === 'checks') { setMode('export'); return }
    if (t.action === 'section') { setError('Use Cut X / Y / Z at the top of the view to section the part.'); return }
    if (!t.kind) return
    const body: Record<string, unknown> = { kind: t.kind }
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
    else if (kind === 'sketch') { setSketchId(id); setMode('sketch') }
  }

  const switchMode = (m: Mode) => {
    if (m === 'sketch' && !sketchId) {
      const s = lastSketch()
      if (!s) { setSketchMenu(true); setMode('part'); return }
      setSketchId(s)
    }
    setMode(m)
    setPicking(null)
  }

  const reopen = () => { setSelected(null); setDialog(null); setMode('part'); setSketchId(null); return run(loadAll) }

  const feature = dialog?.type === 'feature' ? tree?.features.find((f) => f.id === dialog.id) : undefined
  const spec = feature ? catalog.find((c) => c.kind === feature.kind) : undefined
  const param = dialog?.type === 'param' && dialog.name ? tree?.params.find((p) => p.name === dialog.name) ?? null : null
  const highlight = selected?.startsWith('feature:') ? selected.slice(8) : null
  const failedAt = tree?.failed ? tree.features.find((f) => f.status === 'fail') : undefined
  const context = selected ? selected.replace(':', ' ') : null

  if (!tree) return <div className="loading">{error ? `Cannot reach the app's backend: ${error}. Is ./dev.sh running?` : 'Loading…'}</div>

  return (
    <div className={`app mode-${mode}`}>
      <Header tree={tree} mode={mode} onMode={switchMode} onUpdate={update} updating={updating} designs={designs}
        onUpdateMode={(m) => { run(() => api.settings({ update_mode: m })).then(() => api.tree().then(setTree)) }}
        onOpen={(slug) => { run(() => api.openProject(slug)).then(reopen) }}
        onNew={() => {
          const name = window.prompt('Name of the new design')
          if (name) run(() => api.createProject(name)).then(reopen)
        }} />

      {mode === 'sketch' ? <div ref={setToolbarHost} className="toolbar-host" /> : (
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

      <main className="main">
        <aside className="side left" aria-label={mode === 'export' ? 'Print and export' : 'Specification tree'}>
          {mode === 'part' && <SpecTree tree={tree} selected={selected} onSelect={setSelected} onEdit={onEdit}
            onRollback={(i) => { run(() => api.rollback(i)).then(changed) }} />}
          {mode === 'sketch' && <div ref={setTreeHost} />}
          {mode === 'export' && <ExportPane tree={tree} checks={checks} printers={printers} printer={project?.settings.printer ?? ''}
            onPrinter={(id) => { run(() => api.settings({ printer: id })).then(() => api.project().then(setProject)).then(() => api.checks().then(setChecks)) }} />}
          {mode === 'part' && (
            <div className="legend">
              <span><Icon name="bolt" size={12} color="var(--stale)" fill="var(--stale)" /> needs update</span>
              <span><Icon name="ok" size={12} color="var(--ok)" width={2.4} /> up to date</span>
              <span><Icon name="warn" size={12} color="var(--warn)" width={2.2} /> check fails</span>
              <span><Icon name="fail" size={12} color="var(--bad)" width={2.4} /> failed</span>
            </div>
          )}
        </aside>

        {mode === 'sketch' && sketchId ? (
          <section className="center sketch-center">
            <SketchEditor key={sketchId} sid={sketchId} tree={tree} params={tree.params} toolbarHost={toolbarHost} treeHost={treeHost}
              onChanged={setTree} onExit={() => { setMode('part'); setSketchId(null) }} />
          </section>
        ) : (
          <ViewportPane part={part} stale={tree.pending_count} failed={failedAt ? `${failedAt.id}: ${failedAt.error}` : null}
            highlight={highlight} picking={dialog?.type === 'feature' ? picking : null} onPick={onPick} onUpdate={update}
            edgeMarks={feature?.edges.length ? feature.edges.map((e) => e.points) : null}>
            {feature && spec && (
              <FeatureDialog feature={feature} spec={spec} tree={tree} fonts={fonts} picking={picking} onPick={setPicking}
                onApply={(patch) => { run(() => api.editFeature(feature.id, patch)).then(changed) }}
                onDelete={() => { run(() => api.deleteFeature(feature.id)).then((r) => { changed(r); if (r) { setDialog(null); setSelected(null) } }) }}
                onEditSketch={(sid) => { setSketchId(sid); setMode('sketch') }}
                onClose={() => { setDialog(null); setPicking(null) }} />
            )}
            {dialog?.type === 'param' && (
              <ParamDialog key={dialog.name ?? 'new'} param={param} tree={tree}
                onSave={(p) => { run(() => api.setParameter(p)).then((r) => { changed(r); if (r && !param) setDialog({ type: 'param', name: p.name }) }) }}
                onDelete={() => { run(() => api.deleteParameter(dialog.name!)).then((r) => { changed(r); if (r) setDialog(null) }) }}
                onClose={() => setDialog(null)} />
            )}
          </ViewportPane>
        )}

        <AiPanel status={ai} activity={activity} context={context}
          onMode={(m) => { run(() => api.settings({ ai_mode: m })).then(() => api.aiStatus().then(setAi)) }}
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
