// The 3D view: the last good build, one mesh per body. While the model is out of date (or a feature failed) it dims and
// says so. Clicks select faces, or pick faces / edges for the feature being edited.
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { api, type PartState, type PlateMesh, type Reference } from '../api'
import { Icon } from '../ui/icons'
import { Viewer, type FacePick, type ViewName } from '../viewer/Viewer'

const BODY_COLORS = [0xcfc8b8, 0xc9a227, 0x8fb8de, 0xc07a45, 0x9fd18c]
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
  children?: React.ReactNode // dialogs float over the view
}

export function ViewportPane({ part, references, stale, failed, highlight, edgeMarks, picking, onPick, onUpdate, plateMode = false, plate = null, children }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const viewer = useRef<Viewer | null>(null)
  const fitted = useRef(false)
  const onPickRef = useRef(onPick)
  onPickRef.current = onPick
  const [section, setSection] = useState<'X' | 'Y' | 'Z' | null>(null)
  const [ortho, setOrtho] = useState(false)

  useEffect(() => {
    const v = new Viewer(ref.current!)
    v.onSelect = (p) => onPickRef.current(p)
    viewer.current = v
    return () => { v.dispose(); viewer.current = null }
  }, [])

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

  const dim = stale > 0 || !!failed
  const view = (n: ViewName) => () => viewer.current?.setView(n)
  return (
    <section className={`vp${dim ? ' stale' : ''}${picking ? ' picking' : ''}`} ref={ref} aria-label="3D view">
      <div className="vtools">
        <div className="grp">
          <button onClick={view('iso')}>Iso</button><button onClick={view('front')}>Front</button>
          <button onClick={view('top')}>Top</button><button onClick={view('right')}>Right</button>
          <button onClick={() => viewer.current?.fit()}>Fit</button>
          <button aria-pressed={ortho} onClick={() => setOrtho(!ortho)}>Ortho</button>
        </div>
        <div className="grp" role="group" aria-label="Section">
          {(['X', 'Y', 'Z'] as const).map((a) => (
            <button key={a} aria-pressed={section === a} onClick={() => setSection(section === a ? null : a)}>Cut {a}</button>
          ))}
        </div>
      </div>
      {dim && (
        <div className={`banner${failed ? ' bad' : ''}`} role="status">
          <Icon name={failed ? 'fail' : 'bolt'} size={15} fill={failed ? 'none' : 'currentColor'} />
          <span>{failed ? `Update stopped: ${failed} · showing the last good part` : `Out of date · ${stale} to update · showing the last update`}</span>
          {stale > 0 && <button className="btn primary sm" onClick={onUpdate}>Update</button>}
        </div>
      )}
      {picking && <div className="pickhint">Picking {picking}: click on the part</div>}
      {!plateMode && part && !part.bodies.length && !references.length && <div className="empty">Nothing built yet: start with a sketch, then extrude it — or ask the AI.</div>}
      {children}
      <div className="vinfo">{plateMode ? (plate ? `Plate ${plate.plate} · ${plate.objects.length} object${plate.objects.length === 1 ? '' : 's'}` : 'Laying out…')
        : part ? `${(part.volume / 1000).toFixed(1)} cm³ · ${(part.mass * 1000).toFixed(0)} g` : ''}</div>
    </section>
  )
}
