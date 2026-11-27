// The 3D view: the last good build, one mesh per body. While the model is out of date (or a feature failed) it dims and
// says so. Clicks select faces, pick faces/edges for features, or smart-fill / brush-paint Character Mode sculpts.
import { useEffect, useId, useRef, useState } from 'react'
import * as THREE from 'three'
import { api, type PartState, type PlateMesh, type Reference, type SculptPaintState, type Vec3 } from '../api'
import { Icon } from '../ui/icons'
import { Spinner } from '../ui/Spinner'
import { Viewer, type FacePick, type ViewName } from '../viewer/Viewer'

const BODY_COLORS = [0xcfc8b8, 0xc9a227, 0x8fb8de, 0xc07a45, 0x9fd18c]
const DEFAULT_PALETTE: Record<string, string> = {
  '1': '#527292',
  '2': '#ece6d6',
  '3': '#f5f7fa',
  '4': '#1d2128',
  '5': '#cc3b3b',
}
const hex = (c: string | null, i: number) => (c && /^#[0-9a-f]{6}$/i.test(c) ? parseInt(c.slice(1), 16) : BODY_COLORS[i % BODY_COLORS.length])

interface Props {
  part: PartState | null
  references: Reference[]
  stale: number // features waiting for an update
  failed: string | null // why the last update stopped
  highlight: string | null // feature id whose faces to tint
  edgeMarks: number[][][] | null
  picking: 'faces' | 'edges' | null
  onPick: (pick: FacePick | null) => void
  onUpdate: () => void
  plateMode?: boolean // the Plates tab: show a print plate instead of the assembly
  plate?: PlateMesh | null // the plate to show (null while it loads)
  hasDesign?: boolean
  hasPhotos?: boolean
  canvasView?: 'model' | 'design' | 'photos'
  onCanvasView?: (v: 'model' | 'design' | 'photos') => void
  activeSculptId?: string | null
  texPainterReady?: boolean
  onPartUpdated?: (part: PartState) => void
  onAiPaint?: (sid: string) => void
  children?: React.ReactNode // dialogs float over the view
}

export function ViewportPane({
  part, references, stale, failed, highlight, edgeMarks, picking, onPick, onUpdate,
  plateMode = false, plate = null, hasDesign = false, hasPhotos = false, canvasView = 'model', onCanvasView,
  activeSculptId = null, texPainterReady = false, onPartUpdated, onAiPaint, children,
}: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const viewer = useRef<Viewer | null>(null)
  const paintBarRef = useRef<HTMLDivElement>(null)
  const fitted = useRef(false)
  const onPickRef = useRef(onPick)
  onPickRef.current = onPick
  const [section, setSection] = useState<'X' | 'Y' | 'Z' | null>(null)
  const [ortho, setOrtho] = useState(false)

  // Character Mode 3D Smart-Fill / Brush Paint state
  const [paintOpen, setPaintOpen] = useState(false)
  const [paintTool, setPaintTool] = useState<'smart_fill' | 'brush' | 'bucket'>('smart_fill')
  const [paintSlot, setPaintSlot] = useState<number>(3)
  const [creaseDeg, setCreaseDeg] = useState<number>(26)
  const [radiusMm, setRadiusMm] = useState<number>(4.0)
  const [paintState, setPaintState] = useState<SculptPaintState | null>(null)
  const [paintingBusy, setPaintingBusy] = useState(false)
  const [paintStatus, setPaintStatus] = useState('')
  const sliderId = useId()
  const colorPickerId = useId()

  const canPaint = Boolean(activeSculptId) && !plateMode && canvasView === 'model'
  const paintActive = paintOpen && canPaint

  // Keep latest paint config accessible inside the Three.js click callback
  const paintCfgRef = useRef({ activeSculptId, paintTool, paintSlot, creaseDeg, radiusMm, onPartUpdated })
  paintCfgRef.current = { activeSculptId, paintTool, paintSlot, creaseDeg, radiusMm, onPartUpdated }
  const paintBusyRef = useRef(false)

  useEffect(() => {
    const v = new Viewer(ref.current!)
    v.onSelect = (p) => onPickRef.current(p)
    v.onPaintPoint = async (pick) => {
      const { activeSculptId: sid, paintTool: mode, paintSlot: filament, creaseDeg: cd, radiusMm: rm, onPartUpdated: notify } = paintCfgRef.current
      if (!sid || !pick.point || paintBusyRef.current) return
      paintBusyRef.current = true
      setPaintingBusy(true)
      try {
        const res = await api.sculptPaintStroke(sid, {
          mode,
          hit_point: pick.point as Vec3,
          filament,
          crease_deg: cd,
          radius_mm: rm,
        })
        setPaintState(res.paint)
        notify?.(res.part)
        const n = res.paint.painted_triangles ?? 0
        setPaintStatus(n > 0 ? `Painted ${n.toLocaleString()} triangles with Slot ${filament}` : 'No triangles changed')
      } catch (err) {
        setPaintStatus(err instanceof Error ? err.message : 'Paint failed')
      } finally {
        paintBusyRef.current = false
        setPaintingBusy(false)
      }
    }
    viewer.current = v
    return () => { v.dispose(); viewer.current = null }
  }, [])

  useEffect(() => {
    if (viewer.current) viewer.current.paintMode = paintActive
  }, [paintActive])

  // Fetch current sculpt paint palette when Paint toolbar opens or sculpt changes
  useEffect(() => {
    if (!paintActive || !activeSculptId) return
    let live = true
    api.sculptPaint(activeSculptId).then((st) => { if (live) setPaintState(st) }).catch(() => {})
    return () => { live = false }
  }, [paintActive, activeSculptId, part?.rev])

  // Roving-tabindex keyboard fallback for the Paint toolbar ONLY when native `focusgroup` is unsupported
  useEffect(() => {
    const bar = paintBarRef.current
    if (!bar || !paintActive || 'focusGroup' in HTMLElement.prototype) return
    const items = () => [...bar.querySelectorAll<HTMLElement>('button, input')]
    const list = items()
    list.forEach((el, i) => { el.tabIndex = i === 0 ? 0 : -1 })
    const onFocusIn = (e: FocusEvent) => {
      const target = e.target as HTMLElement | null
      const cur = items()
      if (target && cur.includes(target)) cur.forEach((el) => { el.tabIndex = el === target ? 0 : -1 })
    }
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target instanceof HTMLInputElement && target.type === 'range') return
      const cur = items()
      const idx = target ? cur.indexOf(target) : -1
      if (idx < 0) return
      const rtl = getComputedStyle(bar).direction === 'rtl'
      let next = -1
      if (e.key === 'ArrowRight') next = (idx + (rtl ? -1 : 1) + cur.length) % cur.length
      else if (e.key === 'ArrowLeft') next = (idx + (rtl ? 1 : -1) + cur.length) % cur.length
      else if (e.key === 'Home') next = 0
      else if (e.key === 'End') next = cur.length - 1
      if (next >= 0) {
        e.preventDefault()
        cur[next]?.focus()
      }
    }
    bar.addEventListener('focusin', onFocusIn)
    bar.addEventListener('keydown', onKeyDown)
    return () => {
      bar.removeEventListener('focusin', onFocusIn)
      bar.removeEventListener('keydown', onKeyDown)
    }
  }, [paintActive, paintTool])

  // Bodies: replace the meshes whenever a new build arrives.
  useEffect(() => {
    const v = viewer.current
    if (!v || !part) return
    const ids = new Set(part.bodies.map((b) => b.id))
    for (const id of v.partNames()) if (!ids.has(id)) v.removePart(id)
    part.bodies.forEach((b, i) => v.addPart({ ...b.mesh, name: b.id }, { color: hex(b.color, i), cap: { fill: 0xe3c79a, line: 0x6b4d1e } }))
    if (!fitted.current && part.bodies.length) {
      fitted.current = true
      setTimeout(() => viewer.current?.fit(), 120) // after the layout settles
    }
  }, [part])

  // References: fetch a reference's meshes when it is added or placed differently; visibility needs no fetch.
  const loaded = useRef(new Map<string, string>())
  useEffect(() => {
    const v = viewer.current
    if (!v) return
    const ids = new Set(references.map((r) => r.id))
    for (const id of v.referenceIds()) if (!ids.has(id)) { v.removeReference(id); loaded.current.delete(id) }
    for (const r of references) {
      const key = JSON.stringify([r.file, r.plate, r.place, r.x, r.y, r.z, r.rotate_z, r.scale, r.color])
      if (loaded.current.get(r.id) === key) { v.setReferenceVisible(r.id, r.visible); continue }
      loaded.current.set(r.id, key)
      api.referenceMesh(r.id).then((m) => {
        if (loaded.current.get(r.id) !== key || !viewer.current) return // placed again meanwhile
        viewer.current.setReference(r.id, m.parts, m.color, r.visible)
        if (!fitted.current) { fitted.current = true; viewer.current.fit() } // nothing else in view yet: show it
      }).catch(() => loaded.current.delete(r.id))
    }
  }, [references])

  useEffect(() => {
    const v = viewer.current
    if (!v || !part) return
    v.setHighlight(null)
    if (!highlight) return
    for (const b of part.bodies) {
      const faces = (b.mesh.face_names ?? []).flatMap((n, i) => (n && (n === highlight || n.startsWith(highlight + '.')) ? [i] : []))
      if (faces.length) { v.setHighlight(b.id, faces); break }
    }
  }, [highlight, part])

  // Plates tab: the assembly steps aside for one print plate; the view frames each plate when it first shows.
  const plateShown = useRef<number | null>(null)
  useEffect(() => {
    const v = viewer.current
    if (!v) return
    v.setPlateMode(plateMode)
    if (!plateMode) { plateShown.current = null; v.fit() }
  }, [plateMode])
  useEffect(() => {
    const v = viewer.current
    if (!v || !plateMode) return
    v.setPlate(plate)
    if (plate && plateShown.current !== plate.plate) { plateShown.current = plate.plate; v.setView('iso'); v.fit() }
  }, [plate, plateMode])

  useEffect(() => { viewer.current?.setEdgeMarks(edgeMarks) }, [edgeMarks])
  useEffect(() => { if (viewer.current) viewer.current.pickMode = picking === 'edges' ? 'edge' : 'face' }, [picking])
  useEffect(() => { viewer.current?.setOrtho(ortho) }, [ortho])
  useEffect(() => {
    const v = viewer.current
    if (!v) return
    v.setSection(!!section)
    if (section) {
      const b = v.bounds()
      const n = section === 'X' ? new THREE.Vector3(1, 0, 0) : section === 'Y' ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1)
      v.setSectionPlane(n, b.center)
      v.onSectionDrag = () => {}
    }
  }, [section, part])

  const palette = { ...DEFAULT_PALETTE, ...(paintState?.palette ?? {}) }
  const activeHex = palette[String(paintSlot)] ?? DEFAULT_PALETTE[String(paintSlot)] ?? '#527292'

  async function onSlotColorChange(nextHex: string) {
    if (!activeSculptId) return
    setPaintState((prev) => (prev ? { ...prev, palette: { ...prev.palette, [String(paintSlot)]: nextHex } } : prev))
    try {
      const res = await api.sculptPaintPalette(activeSculptId, { [String(paintSlot)]: nextHex })
      setPaintState(res.paint)
      onPartUpdated?.(res.part)
      setPaintStatus(`Updated Slot ${paintSlot} color to ${nextHex}`)
    } catch { /* ignore */ }
  }

  async function onPaintUndo() {
    if (!activeSculptId || !paintState?.can_undo || paintingBusy) return
    setPaintingBusy(true)
    try {
      const res = await api.sculptPaintUndo(activeSculptId)
      setPaintState(res.paint)
      onPartUpdated?.(res.part)
      setPaintStatus('Undid last paint stroke')
    } finally {
      setPaintingBusy(false)
    }
  }

  async function onPaintReset() {
    if (!activeSculptId || paintingBusy) return
    setPaintingBusy(true)
    try {
      const res = await api.sculptPaintReset(activeSculptId)
      setPaintState(res.paint)
      onPartUpdated?.(res.part)
      setPaintStatus('Reset paint to default colors')
    } finally {
      setPaintingBusy(false)
    }
  }

  const dim = stale > 0 || !!failed
  const view = (n: ViewName) => () => viewer.current?.setView(n)
  return (
    <section className={`vp${dim ? ' stale' : ''}${picking || paintActive ? ' picking' : ''}`} ref={ref} aria-label="3D view">
      <div className="vtools">
        {(hasDesign || hasPhotos) && onCanvasView && (
          <div className="grp" role="group" aria-label="Canvas view">
            <button type="button" aria-pressed={canvasView === 'model'} onClick={() => onCanvasView('model')}>Model</button>
            {hasDesign && (
              <button type="button" aria-pressed={canvasView === 'design'} onClick={() => onCanvasView('design')}>Design</button>
            )}
            {hasPhotos && (
              <button type="button" aria-pressed={canvasView === 'photos'} onClick={() => onCanvasView('photos')}>Photos</button>
            )}
          </div>
        )}
        {canPaint && (
          <div className="grp" role="group" aria-label="3D Paint">
            <button type="button" aria-pressed={paintOpen} onClick={() => setPaintOpen((v) => !v)} title="Paint filament colors directly onto 3D eyes, teeth, and accents">
              Paint
            </button>
          </div>
        )}
        <span className="spacer" />
        <div className="grp">
          <button type="button" onClick={view('iso')}>Iso</button><button type="button" onClick={view('front')}>Front</button>
          <button type="button" onClick={view('top')}>Top</button><button type="button" onClick={view('right')}>Right</button>
          <button type="button" onClick={() => viewer.current?.fit()}>Fit</button>
          <button type="button" aria-pressed={ortho} onClick={() => setOrtho(!ortho)}>Ortho</button>
        </div>
        <div className="grp" role="group" aria-label="Section">
          {(['X', 'Y', 'Z'] as const).map((a) => (
            <button type="button" key={a} aria-pressed={section === a} onClick={() => setSection(section === a ? null : a)}>Cut {a}</button>
          ))}
        </div>
      </div>
      {dim && (
        <div className={`banner${failed ? ' bad' : ''}`} role="status">
          <Icon name={failed ? 'fail' : 'bolt'} size={15} fill={failed ? 'none' : 'currentColor'} />
          <span>{failed ? `Update stopped: ${failed} · showing the last good part` : `Out of date · ${stale} to update · showing the last update`}</span>
          {stale > 0 && <button type="button" className="btn primary sm" onClick={onUpdate}>Update</button>}
        </div>
      )}
      {picking && <div className="pickhint">Picking {picking}: click on the part</div>}
      {paintActive && !picking && (
        <>
          <div className="pickhint">
            {paintTool === 'smart_fill'
              ? 'Smart Fill: click any eye, tooth, tongue, or region to fill up to its 3D crease (drag to orbit)'
              : paintTool === 'brush'
                ? 'Brush: click on the sculpt to dab filament color (drag to orbit)'
                : 'Bucket: click any colored region to recolor it (drag to orbit)'}
          </div>
          <div
            ref={paintBarRef}
            className="paint-bar"
            role="toolbar"
            aria-label="3D Paint tools"
            {...({ focusgroup: 'toolbar wrap' } as Record<string, string>)}
          >
            <div className="paint-grp" role="group" aria-label="Paint mode">
              <button type="button" aria-pressed={paintTool === 'smart_fill'} onClick={() => setPaintTool('smart_fill')} title="1-click 3D crease-guided smart fill">
                Smart Fill
              </button>
              <button type="button" aria-pressed={paintTool === 'brush'} onClick={() => setPaintTool('brush')} title="Circular surface brush">
                Brush
              </button>
              <button type="button" aria-pressed={paintTool === 'bucket'} onClick={() => setPaintTool('bucket')} title="Recolor an entire painted region">
                Bucket
              </button>
            </div>
            <span className="paint-sep" role="separator" aria-orientation="vertical" />
            <div className="paint-grp swatches" role="group" aria-label="Filament slots">
              {[1, 2, 3, 4, 5].map((s) => {
                const col = palette[String(s)] ?? DEFAULT_PALETTE[String(s)]
                return (
                  <button
                    key={s}
                    type="button"
                    className="swatch"
                    aria-pressed={paintSlot === s}
                    aria-label={`Extruder ${s} (${col})`}
                    title={`Extruder ${s} (${col})`}
                    style={{ '--sw': col } as React.CSSProperties}
                    onClick={() => setPaintSlot(s)}
                  >
                    {s}
                  </button>
                )
              })}
              <label className="color-pick" htmlFor={colorPickerId} title={`Change Extruder ${paintSlot} color`}>
                <span className="sr-only">Extruder {paintSlot} color</span>
                <input
                  id={colorPickerId}
                  type="color"
                  value={activeHex}
                  onChange={(e) => onSlotColorChange(e.target.value)}
                />
              </label>
            </div>
            {(paintTool === 'smart_fill' || paintTool === 'brush') && (
              <>
                <span className="paint-sep" role="separator" aria-orientation="vertical" />
                <div className="paint-slider-wrap" role="group" aria-label={paintTool === 'smart_fill' ? 'Crease angle' : 'Brush radius'}>
                  <label htmlFor={sliderId}>
                    {paintTool === 'smart_fill' ? `Crease ${creaseDeg}°` : `Size ${radiusMm.toFixed(1)} mm`}
                  </label>
                  <input
                    id={sliderId}
                    className="paint-slider"
                    type="range"
                    min={paintTool === 'smart_fill' ? 10 : 1}
                    max={paintTool === 'smart_fill' ? 60 : 18}
                    step={paintTool === 'smart_fill' ? 2 : 0.5}
                    value={paintTool === 'smart_fill' ? creaseDeg : radiusMm}
                    aria-valuetext={paintTool === 'smart_fill' ? `${creaseDeg} degrees` : `${radiusMm.toFixed(1)} millimetres`}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value)
                      if (paintTool === 'smart_fill') setCreaseDeg(v)
                      else setRadiusMm(v)
                    }}
                  />
                </div>
              </>
            )}
            <span className="paint-sep" role="separator" aria-orientation="vertical" />
            <div className="paint-grp" role="group" aria-label="Paint actions">
              <button
                type="button"
                aria-disabled={!paintState?.can_undo || paintingBusy}
                onClick={onPaintUndo}
                title="Undo last paint stroke"
              >
                Undo
              </button>
              <button
                type="button"
                aria-disabled={paintingBusy}
                onClick={onPaintReset}
                title="Reset to automatic colors"
              >
                Reset
              </button>
              {texPainterReady && onAiPaint && activeSculptId && (
                <button
                  type="button"
                  aria-disabled={paintingBusy}
                  onClick={() => { if (!paintingBusy) onAiPaint(activeSculptId) }}
                  title="Bake full 3D colors from the approved concept image using local Hunyuan3D-Paint"
                >
                  Auto 3D Paint
                </button>
              )}
            </div>
          </div>
          <p className="sr-only" role="status">{paintStatus}</p>
        </>
      )}
      {!plateMode && part && !part.bodies.length && !references.length && <div className="empty">Nothing built yet: start with a sketch, then extrude it — or ask the AI.</div>}
      {children}
      <div className="vinfo">{plateMode
        ? (plate ? `Plate ${plate.plate} · ${plate.objects.length} object${plate.objects.length === 1 ? '' : 's'}` : <span className="spin-row"><Spinner size={12} decorative /> Laying out…</span>)
        : part ? `${(part.volume / 1000).toFixed(1)} cm³ · ${(part.mass * 1000).toFixed(0)} g` : ''}</div>
    </section>
  )
}
