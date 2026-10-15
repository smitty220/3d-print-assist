import * as THREE from 'three'
import type { Fastener, Finding, IntentObject, InterpretOverlay, ShapeMesh, TopoPreview } from '../api'
import { buildFindingMarker } from './findingMarker'
import { buildOverlay } from './intentOverlay'
import { buildTopoPreview } from './topoPreview'
import { SectionTool, type CapStyle } from './section'

// Traced outlines: the profile, through cut-outs, and pockets from the top and the bottom.
export const SKETCH_COLORS = [0x1d2733, 0xe0782a, 0x2f9e8f, 0x8a4fd6, 0xc2412d, 0x2a6fd9]

export type MouseScheme = 'catia' | 'simple'
export type ViewName = 'iso' | 'front' | 'top' | 'right'

export interface FacePick {
  part: string
  faceId: number
  /** Set in edge-picking mode: the B-rep edge under the cursor (faceId is then -1). */
  edgeId?: number
  /** Where the ray hit the part, world mm. */
  point?: [number, number, number]
}

export type PickMode = 'face' | 'edge'

/** A surface mesh for display: positions (mm), triangles, and element edges as index pairs. */
export interface SurfaceMesh {
  positions: number[]
  indices: number[]
  edges: number[] | null
}

/** Mesh controls drawn on a part: edges with divisions, mapped faces, spheres of influence. `active` is highlighted. */
export interface MeshControlOverlay {
  part: string
  edges: { id: number; n: number }[]
  faces: number[]
  spheres: { center: number[]; radius: number }[]
  active: { kind: 'edge' | 'face' | 'sphere'; index: number } | null
}

export interface PartStyle {
  color: number
  opacity?: number
  pickable?: boolean
  cap: Omit<CapStyle, 'angle'>
}

export interface SceneBounds {
  center: THREE.Vector3
  radius: number
}

interface PartEntry {
  data: ShapeMesh
  group: THREE.Group
  mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>
  edges: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>
  base: THREE.Color
  faceRanges: Map<number, [start: number, count: number]>
  faceOfVertex: Uint32Array
  edgeOfSegment: Uint32Array
}

const HOVER = new THREE.Color(0xffc266)
const SELECT = new THREE.Color(0xff9a1f)
const HIGHLIGHT = new THREE.Color(0xf2b25c)
const FOV = 35
const VIEWS: Record<ViewName, [theta: number, phi: number]> = {
  iso: [-Math.PI / 4 - 0.25, 1.08],
  front: [-Math.PI / 2, Math.PI / 2],
  top: [-Math.PI / 2, 0.001],
  right: [0, Math.PI / 2],
}

/** Three.js viewport with CATIA-style navigation, per-B-rep-face picking and a section tool. Scene units are mm, Z up. */
export class Viewer {
  onSelect?: (pick: FacePick | null) => void
  /** Called while the section handle is dragged, with the plane's movement along its normal in mm. */
  onSectionDrag?: (deltaMm: number) => void
  mouseScheme: MouseScheme = 'catia'
  /** What a click picks: B-rep faces (default) or edges (for mesh controls). */
  pickMode: PickMode = 'face'

  private el: HTMLElement
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private persp = new THREE.PerspectiveCamera(FOV, 1, 0.1, 1e5)
  private ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -1e5, 1e5)
  private cam: THREE.Camera = this.persp
  private orbit = { target: new THREE.Vector3(), r: 400, theta: VIEWS.iso[0], phi: VIEWS.iso[1] }
  private parts = new Map<string, PartEntry>()
  private hover: FacePick | null = null
  private selected: FacePick | null = null
  private highlight: { part: string; faces: Set<number> } | null = null
  private raycaster = new THREE.Raycaster()
  private triScene = new THREE.Scene()
  private triCam = new THREE.PerspectiveCamera(35, 1, 0.1, 20)
  private section: SectionTool
  private sectionOn = false
  private intentGroup = new THREE.Group()
  private findingGroup = new THREE.Group()
  private topoGroup = new THREE.Group()
  private traceGroup = new THREE.Group()
  private contourGroup = new THREE.Group()
  private meshGroup = new THREE.Group()
  private controlGroup = new THREE.Group()
  private edgeHoverGroup = new THREE.Group()
  private edgeMarkGroup = new THREE.Group()
  private abort = new AbortController()
  private frame = 0
  private resizeObs: ResizeObserver
  /** The flag at the contour's peak: an HTML label over the canvas, kept on the projected point each frame. */
  private peakFlag: { el: HTMLDivElement; at: THREE.Vector3 } | null = null

  constructor(el: HTMLElement) {
    this.el = el
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, stencil: true })
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    this.renderer.setPixelRatio(dpr)
    this.renderer.setClearColor(0x000000, 0)
    this.renderer.localClippingEnabled = true
    el.prepend(this.renderer.domElement)
    for (const c of [this.persp, this.ortho, this.triCam]) c.up.set(0, 0, 1)

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x3c4855, 2.2))
    const key = new THREE.DirectionalLight(0xffffff, 2.4)
    key.position.set(5, -7, 9)
    const fill = new THREE.DirectionalLight(0xffffff, 0.9)
    fill.position.set(-6, 5, 3)
    this.scene.add(key, fill)

    this.section = new SectionTool(dpr)
    this.scene.add(this.section.caps, this.section.gizmo, this.intentGroup, this.findingGroup, this.topoGroup, this.traceGroup, this.contourGroup,
      this.meshGroup, this.controlGroup, this.edgeHoverGroup, this.edgeMarkGroup)

    const axes: [THREE.Vector3, number][] = [
      [new THREE.Vector3(1, 0, 0), 0xd94a3a],
      [new THREE.Vector3(0, 1, 0), 0x3aa856],
      [new THREE.Vector3(0, 0, 1), 0x3a7bd9],
    ]
    for (const [d, c] of axes) this.triScene.add(new THREE.ArrowHelper(d, new THREE.Vector3(), 1, c, 0.32, 0.16))

    this.resizeObs = new ResizeObserver(() => this.resize())
    this.resizeObs.observe(el)
    this.bindInput()
    this.resize()
    this.loop()
  }

  dispose() {
    cancelAnimationFrame(this.frame)
    this.abort.abort()
    this.resizeObs.disconnect()
    for (const name of [...this.parts.keys()]) this.removePart(name)
    this.renderer.dispose()
    this.renderer.domElement.remove()
    this.setPeakFlag(null)
  }

  // ---------- parts ----------

  addPart(data: ShapeMesh, style: PartStyle) {
    this.removePart(data.name)
    const positions: number[] = []
    const indices: number[] = []
    const faceIds: number[] = []
    const faceRanges = new Map<number, [number, number]>()
    for (const f of data.faces) {
      const start = positions.length / 3
      for (const v of f.positions) positions.push(v) // no spread: topology surfaces have ~10⁵ values
      for (const i of f.indices) indices.push(i + start)
      const count = f.positions.length / 3
      for (let i = 0; i < count; i++) faceIds.push(f.id)
      faceRanges.set(f.id, [start, count])
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geo.setIndex(indices)
    geo.computeVertexNormals()
    const base = new THREE.Color(style.color)
    const colors = new Float32Array(positions.length)
    for (let i = 0; i < colors.length; i += 3) base.toArray(colors, i)
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    const opacity = style.opacity ?? 1
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.2,
      roughness: 0.55,
      transparent: opacity < 1,
      opacity,
      depthWrite: opacity >= 1,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    })
    const mesh = new THREE.Mesh(geo, mat)
    if (style.pickable === false) mesh.raycast = NO_RAYCAST

    const seg: number[] = []
    const edgeOfSegment: number[] = []
    for (const e of data.edges) {
      const p = e.points
      for (let i = 0; i + 5 < p.length; i += 3) {
        seg.push(p[i], p[i + 1], p[i + 2], p[i + 3], p[i + 4], p[i + 5])
        edgeOfSegment.push(e.id)
      }
    }
    const edgeGeo = new THREE.BufferGeometry()
    edgeGeo.setAttribute('position', new THREE.Float32BufferAttribute(seg, 3))
    const edges = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0x2a3138, transparent: true, opacity: opacity < 1 ? 0.35 : 0.8 }))
    edges.raycast = () => {}

    const group = new THREE.Group()
    group.add(mesh, edges)
    group.userData.part = data.name
    this.scene.add(group)
    const entry: PartEntry = { data, group, mesh, edges, base, faceRanges, faceOfVertex: Uint32Array.from(faceIds),
      edgeOfSegment: Uint32Array.from(edgeOfSegment) }
    this.parts.set(data.name, entry)
    this.applyClipping(entry)
    if (this.highlight?.part === data.name) for (const f of this.highlight.faces) this.paint({ part: data.name, faceId: f })

    const angle = this.parts.size % 2 ? 45 : 135
    this.section.addPart(data.name, geo, { ...style.cap, angle })
  }

  removePart(name: string) {
    const p = this.parts.get(name)
    if (!p) return
    this.section.removePart(name)
    this.scene.remove(p.group)
    p.mesh.geometry.dispose()
    p.mesh.material.dispose()
    p.edges.geometry.dispose()
    p.edges.material.dispose()
    this.parts.delete(name)
    if (this.selected?.part === name) this.select(null)
    if (this.hover?.part === name) this.hover = null
  }

  setVisible(name: string, visible: boolean) {
    const p = this.parts.get(name)
    if (!p) return
    p.group.visible = visible
    this.section.setPartVisible(name, visible)
  }

  bounds(): SceneBounds {
    const box = new THREE.Box3()
    for (const p of this.parts.values()) if (p.group.visible) box.expandByObject(p.group)
    if (box.isEmpty()) return { center: new THREE.Vector3(), radius: 100 }
    const s = box.getBoundingSphere(new THREE.Sphere())
    return { center: s.center, radius: s.radius }
  }

  /** Unit vector the camera is looking along. */
  viewDir() {
    return this.orbit.target.clone().sub(this.persp.position).normalize()
  }

  // ---------- intent overlays ----------

  setIntent(objects: IntentObject[], catalog: Map<string, Fastener>, selectedId: string | null) {
    for (const child of [...this.intentGroup.children]) {
      this.intentGroup.remove(child)
      child.traverse((n) => {
        const m = n as THREE.Mesh
        m.geometry?.dispose()
        ;(m.material as THREE.Material | undefined)?.dispose()
      })
    }
    for (const o of objects) {
      const g = buildOverlay(o, objects, catalog, o.id === selectedId)
      if (g) this.intentGroup.add(g)
    }
    this.applyOverlayClipping()
  }

  /** Show one validation finding's location (or nothing). */
  setFinding(f: Finding | null) {
    for (const child of [...this.findingGroup.children]) {
      this.findingGroup.remove(child)
      child.traverse((n) => {
        const m = n as THREE.Mesh
        m.geometry?.dispose()
        ;(m.material as THREE.Material | undefined)?.dispose()
      })
    }
    const g = f && buildFindingMarker(f, Math.max(this.bounds().radius * 0.025, 0.5))
    if (g) this.findingGroup.add(g)
  }

  /** Show the topology set-up (interface elements, supports, loads), or clear it. */
  setTopoPreview(p: TopoPreview | null) {
    clearGroup(this.topoGroup)
    if (p) this.topoGroup.add(buildTopoPreview(p, Math.max(this.bounds().radius * 0.35, 5)))
    this.applyOverlayClipping()
  }

  /** A stress (or displacement) contour on the part's surface, coloured from `lo` (blue) to `hi` (red): a smooth gradient,
   *  or `bands` flat colour bands (each a fixed range of the value). `edges` (index pairs) draws the element edges over
   *  it. A marker sits at the peak. Pass null to clear. */
  setContour(c: { positions: number[]; indices: number[]; values: number[]; lo: number; hi: number; peak: number[]; bands?: number; edges?: number[] | null; peakLabel?: string } | null) {
    clearGroup(this.contourGroup)
    this.setPeakFlag(null)
    if (!c) return
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(c.positions, 3))
    geo.setIndex(c.indices)
    // The value as a texture coordinate into a colour-map strip: interpolated per pixel, so bands come out crisp.
    const uv = new Float32Array(c.values.length * 2)
    const span = c.hi - c.lo || 1
    c.values.forEach((v, i) => {
      uv[2 * i] = Math.min(1, Math.max(0, (v - c.lo) / span))
      uv[2 * i + 1] = 0.5
    })
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2))
    geo.computeVertexNormals()
    const map = colorStrip(c.bands ?? 0)
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map, roughness: 0.75, metalness: 0, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }))
    this.contourGroup.add(mesh)
    if (c.edges?.length) this.contourGroup.add(edgeLines(geo.getAttribute('position') as THREE.BufferAttribute, c.edges, 0x111111, 0.45))
    const r = Math.max(this.bounds().radius * 0.012, 0.4)
    const peak = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), new THREE.MeshBasicMaterial({ color: 0x111111, depthTest: false }))
    peak.position.set(c.peak[0], c.peak[1], c.peak[2])
    peak.renderOrder = 6
    this.contourGroup.add(peak)
    if (c.peakLabel) this.setPeakFlag(peak.position.clone(), c.peakLabel)
    this.applyOverlayClipping()
  }

  private setPeakFlag(at: THREE.Vector3 | null, text = '') {
    this.peakFlag?.el.remove()
    this.peakFlag = null
    if (!at) return
    if (getComputedStyle(this.el).position === 'static') this.el.style.position = 'relative'
    const el = document.createElement('div')
    el.className = 'peak-flag'
    el.setAttribute('role', 'note')
    const label = document.createElement('span')
    label.textContent = text
    el.append(label)
    this.el.append(el)
    this.peakFlag = { el, at }
  }

  /** Keep the peak flag on its point: the flag's tail corner (bottom left) sits at the projected peak. */
  private placePeakFlag(w: number, h: number) {
    const f = this.peakFlag
    if (!f) return
    const p = f.at.clone().project(this.cam)
    const off = p.z > 1 || p.z < -1 || Math.abs(p.x) > 1.05 || Math.abs(p.y) > 1.05
    f.el.style.display = off ? 'none' : ''
    if (off) return
    f.el.style.transform = `translate(${((p.x + 1) / 2) * w}px, ${((1 - p.y) / 2) * h}px)`
  }

  /** The finite-element mesh on its own (before solving): a light surface with the element edges. Null clears it. */
  setMeshPreview(m: SurfaceMesh | null) {
    clearGroup(this.meshGroup)
    if (!m) return
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(m.positions, 3))
    geo.setIndex(m.indices)
    geo.computeVertexNormals()
    this.meshGroup.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xc9d3dc, roughness: 0.8, metalness: 0, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })))
    if (m.edges?.length) this.meshGroup.add(edgeLines(geo.getAttribute('position') as THREE.BufferAttribute, m.edges, 0x1f3550, 0.7))
    this.applyOverlayClipping()
  }

  /** Edges picked for a blend (polylines, world mm), drawn over the part; the active list is orange. Null clears. */
  setEdgeMarks(lines: number[][][] | null) {
    clearGroup(this.edgeMarkGroup)
    for (const pts of lines ?? []) {
      if (pts.length < 2) continue
      const geo = new THREE.BufferGeometry().setFromPoints(pts.map((q) => new THREE.Vector3(q[0], q[1], q[2])))
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff9a1f, depthTest: false, transparent: true, linewidth: 2 }))
      line.renderOrder = 7
      this.edgeMarkGroup.add(line)
      const ends = new THREE.BufferGeometry().setFromPoints([pts[0], pts[pts.length - 1]].map((q) => new THREE.Vector3(q[0], q[1], q[2])))
      const dots = new THREE.Points(ends, new THREE.PointsMaterial({ color: 0xff9a1f, size: Math.max(this.bounds().radius * 0.016, 0.5), depthTest: false, transparent: true }))
      dots.renderOrder = 7
      this.edgeMarkGroup.add(dots)
    }
  }

  /** Mark the mesh controls on a part: edges with divisions (with tick marks at each division), mapped faces, spheres. */
  setMeshControls(o: MeshControlOverlay | null) {
    clearGroup(this.controlGroup)
    const p = o ? this.parts.get(o.part) : undefined
    if (!o || !p) return
    const isActive = (kind: 'edge' | 'face' | 'sphere', i: number) => o.active?.kind === kind && o.active.index === i
    const tick = Math.max(this.bounds().radius * 0.008, 0.25)
    o.edges.forEach((e, i) => {
      const poly = p.data.edges.find((x) => x.id === e.id)
      if (!poly) return
      const pts: THREE.Vector3[] = []
      for (let k = 0; k < poly.points.length; k += 3) pts.push(new THREE.Vector3(poly.points[k], poly.points[k + 1], poly.points[k + 2]))
      const color = isActive('edge', i) ? 0xff9a1f : 0xd6338a
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true }))
      line.renderOrder = 7
      this.controlGroup.add(line)
      // Division marks: n + 1 dots evenly spaced along the edge.
      const lens = [0]
      for (let k = 1; k < pts.length; k++) lens.push(lens[k - 1] + pts[k].distanceTo(pts[k - 1]))
      const L = lens[lens.length - 1] || 1
      const dots: number[] = []
      for (let j = 0; j <= e.n; j++) {
        const target = (j / e.n) * L
        let k = 1
        while (k < lens.length - 1 && lens[k] < target) k++
        const t = (target - lens[k - 1]) / ((lens[k] - lens[k - 1]) || 1)
        const q = pts.length > 1 ? pts[k - 1].clone().lerp(pts[k], t) : pts[0]
        dots.push(q.x, q.y, q.z)
      }
      const pg = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(dots, 3))
      const marks = new THREE.Points(pg, new THREE.PointsMaterial({ color, size: tick * 2.2, depthTest: false, transparent: true }))
      marks.renderOrder = 7
      this.controlGroup.add(marks)
    })
    o.faces.forEach((fid, i) => {
      const f = p.data.faces.find((x) => x.id === fid)
      if (!f) return
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.Float32BufferAttribute(f.positions, 3))
      geo.setIndex(f.indices)
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: isActive('face', i) ? 0xff9a1f : 0x2f9e8f, transparent: true, opacity: 0.45,
        depthTest: false, side: THREE.DoubleSide }))
      m.renderOrder = 6
      this.controlGroup.add(m)
    })
    o.spheres.forEach((sp, i) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(sp.radius, 28, 18), new THREE.MeshBasicMaterial({
        color: isActive('sphere', i) ? 0xff9a1f : 0x3a7bd9, wireframe: true, transparent: true, opacity: isActive('sphere', i) ? 0.6 : 0.28 }))
      m.position.set(sp.center[0], sp.center[1], sp.center[2])
      this.controlGroup.add(m)
    })
  }

  /** The sketches a part was drafted from, one colour each, drawn over everything. */
  setTraceOverlay(o: InterpretOverlay | null) {
    clearGroup(this.traceGroup)
    if (!o) return
    o.sketches.forEach((s, i) => {
      const material = new THREE.LineBasicMaterial({ color: SKETCH_COLORS[i % SKETCH_COLORS.length], depthTest: false, transparent: true })
      for (const pts of s.polylines) {
        const line = new THREE.Line(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pts, 3)), material)
        line.renderOrder = 5
        this.traceGroup.add(line)
      }
    })
  }

  private applyOverlayClipping() {
    const planes = this.sectionOn ? [this.section.plane] : null
    for (const group of [this.intentGroup, this.topoGroup, this.contourGroup, this.meshGroup]) {
      group.traverse((n) => {
        const m = (n as THREE.Mesh).material as THREE.Material | undefined
        if (m) {
          m.clippingPlanes = planes
          m.needsUpdate = true
        }
      })
    }
  }

  // ---------- section ----------

  setSection(on: boolean) {
    this.sectionOn = on
    this.section.setEnabled(on)
    for (const p of this.parts.values()) this.applyClipping(p)
    this.applyOverlayClipping()
  }

  setSectionPlane(normal: THREE.Vector3, point: THREE.Vector3) {
    const b = this.bounds()
    this.section.setPlane(normal, point, b.center, b.radius)
  }

  setSectionPlaneVisible(show: boolean) {
    this.section.setShowPlane(show)
  }

  private applyClipping(p: PartEntry) {
    const planes = this.sectionOn ? [this.section.plane] : null
    for (const m of [p.mesh.material, p.edges.material]) {
      m.clippingPlanes = planes
      m.needsUpdate = true
    }
  }

  // ---------- selection ----------

  select(pick: FacePick | null) {
    const prev = this.selected
    this.selected = pick
    if (prev) this.paint(prev)
    if (pick) this.paint(pick)
    this.onSelect?.(pick)
  }

  /** Drop the selection and hover without telling onSelect (after a pick that was not a selection). */
  clearPick() {
    const prev = this.selected
    this.selected = null
    if (prev) this.paint(prev)
    this.setHover(null)
  }

  /** Tint a group of faces (e.g. every face a feature produced) without changing the selection. */
  setHighlight(part: string | null, faceIds: number[] = []) {
    const prev = this.highlight
    this.highlight = part ? { part, faces: new Set(faceIds) } : null
    if (prev) for (const f of prev.faces) this.paint({ part: prev.part, faceId: f })
    if (this.highlight) for (const f of this.highlight.faces) this.paint({ part: this.highlight.part, faceId: f })
  }

  private setHover(pick: FacePick | null) {
    if (same(pick, this.hover)) return
    const prev = this.hover
    this.hover = pick
    if (prev) this.paint(prev)
    if (pick) this.paint(pick)
    clearGroup(this.edgeHoverGroup)
    const p = pick?.edgeId != null ? this.parts.get(pick.part) : undefined
    const poly = p?.data.edges.find((e) => e.id === pick!.edgeId)
    if (poly) {
      const line = new THREE.Line(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(poly.points, 3)),
        new THREE.LineBasicMaterial({ color: 0xff9a1f, depthTest: false, transparent: true }))
      line.renderOrder = 8
      this.edgeHoverGroup.add(line)
    }
  }

  private paint(pick: FacePick) {
    const p = this.parts.get(pick.part)
    const range = p?.faceRanges.get(pick.faceId)
    if (!p || !range) return
    const lit = this.highlight?.part === pick.part && this.highlight.faces.has(pick.faceId)
    const color = same(pick, this.selected) ? SELECT : same(pick, this.hover) ? HOVER : lit ? HIGHLIGHT : p.base
    const attr = p.mesh.geometry.getAttribute('color') as THREE.BufferAttribute
    const [start, count] = range
    for (let i = start; i < start + count; i++) attr.setXYZ(i, color.r, color.g, color.b)
    attr.needsUpdate = true
  }

  private setRay(clientX: number, clientY: number) {
    const r = this.renderer.domElement.getBoundingClientRect()
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1)
    this.raycaster.setFromCamera(ndc, this.cam)
  }

  private pickAt(clientX: number, clientY: number): FacePick | null {
    this.setRay(clientX, clientY)
    if (this.pickMode === 'edge') return this.pickEdge(clientX)
    const meshes = [...this.parts.values()].filter((p) => p.group.visible && p.mesh.raycast !== NO_RAYCAST).map((p) => p.mesh)
    for (const hit of this.raycaster.intersectObjects(meshes, false)) {
      if (this.sectionOn && this.section.plane.distanceToPoint(hit.point) < 0) continue
      const entry = [...this.parts.values()].find((p) => p.mesh === hit.object)
      if (entry && hit.face) return { part: entry.data.name, faceId: entry.faceOfVertex[hit.face.a], point: [hit.point.x, hit.point.y, hit.point.z] }
    }
    return null
  }

  /** The B-rep edge nearest the cursor (within a few pixels), on a pickable part, not hidden behind it. */
  private pickEdge(_clientX: number): FacePick | null {
    const px = (2 * this.halfHeight()) / Math.max(1, this.el.clientHeight)
    this.raycaster.params.Line = { threshold: 5 * px }
    const meshes = [...this.parts.values()].filter((p) => p.group.visible && p.mesh.raycast !== NO_RAYCAST)
    const front = this.raycaster.intersectObjects(meshes.map((p) => p.mesh), false)[0]
    let best: { entry: PartEntry; hit: THREE.Intersection } | null = null
    for (const entry of meshes) {
      const hits: THREE.Intersection[] = []
      THREE.LineSegments.prototype.raycast.call(entry.edges, this.raycaster, hits)
      for (const h of hits) {
        if (front && h.distance > front.distance + 8 * px) continue // behind the surface
        if (this.sectionOn && this.section.plane.distanceToPoint(h.point) < 0) continue
        if (!best || (h.distanceToRay ?? 0) < (best.hit.distanceToRay ?? 0)) best = { entry, hit: h }
      }
    }
    if (!best || best.hit.index == null) return null
    const p = best.hit.point
    return { part: best.entry.data.name, faceId: -1, edgeId: best.entry.edgeOfSegment[best.hit.index >> 1], point: [p.x, p.y, p.z] }
  }

  private overHandle(clientX: number, clientY: number) {
    if (!this.sectionOn) return false
    this.setRay(clientX, clientY)
    return this.section.hitsHandle(this.raycaster)
  }

  /** Converts a mouse movement into plane travel along its normal, using the normal's on-screen direction. */
  private handleDrag(dx: number, dy: number) {
    const n = this.section.plane.normal
    const L = this.section.handleLength
    const a = this.section.gizmoCenter.clone().project(this.cam)
    const b = this.section.gizmoCenter.clone().addScaledVector(n, L).project(this.cam)
    const w = this.el.clientWidth
    const h = this.el.clientHeight
    const sx = ((b.x - a.x) * w) / 2
    const sy = (-(b.y - a.y) * h) / 2
    const len2 = sx * sx + sy * sy
    const delta = len2 > 400 ? ((dx * sx + dy * sy) / len2) * L : (-dy * 2 * this.halfHeight()) / h
    this.onSectionDrag?.(delta)
  }

  // ---------- camera ----------

  fit() {
    const b = this.bounds()
    this.orbit.target.copy(b.center)
    this.orbit.r = (b.radius / Math.sin(THREE.MathUtils.degToRad(FOV / 2))) * 1.1
    this.updateCam()
  }

  setView(v: ViewName) {
    ;[this.orbit.theta, this.orbit.phi] = VIEWS[v]
    this.updateCam()
  }

  /** Look straight at the section plane from its kept side's viewing direction. */
  viewNormalToSection() {
    const n = this.section.plane.normal
    this.orbit.phi = Math.acos(THREE.MathUtils.clamp(-n.z, -1, 1))
    this.orbit.phi = THREE.MathUtils.clamp(this.orbit.phi, 0.001, Math.PI - 0.001)
    this.orbit.theta = Math.atan2(-n.y, -n.x)
    this.orbit.target.copy(this.section.gizmoCenter)
    this.updateCam()
  }

  setOrtho(on: boolean) {
    this.cam = on ? this.ortho : this.persp
    this.updateCam()
  }

  private updateCam() {
    const o = this.orbit
    const s = Math.sin(o.phi)
    const off = new THREE.Vector3(o.r * s * Math.cos(o.theta), o.r * s * Math.sin(o.theta), o.r * Math.cos(o.phi))
    const w = this.el.clientWidth || 1
    const h = this.el.clientHeight || 1
    for (const c of [this.persp, this.ortho]) {
      c.position.copy(o.target).add(off)
      c.lookAt(o.target)
    }
    this.persp.aspect = w / h
    this.persp.near = o.r / 200
    this.persp.far = o.r * 50
    this.persp.updateProjectionMatrix()
    const fh = this.halfHeight()
    Object.assign(this.ortho, { left: (-fh * w) / h, right: (fh * w) / h, top: fh, bottom: -fh })
    this.ortho.updateProjectionMatrix()
  }

  private halfHeight() {
    return this.orbit.r * Math.tan(THREE.MathUtils.degToRad(FOV / 2))
  }

  private resize() {
    this.renderer.setSize(this.el.clientWidth, this.el.clientHeight, false)
    this.updateCam()
  }

  private rotate(dx: number, dy: number) {
    this.orbit.theta -= dx * 0.008
    this.orbit.phi = THREE.MathUtils.clamp(this.orbit.phi - dy * 0.008, 0.001, Math.PI - 0.001)
    this.updateCam()
  }

  private pan(dx: number, dy: number) {
    const wpp = (2 * this.halfHeight()) / (this.el.clientHeight || 1)
    const right = new THREE.Vector3().setFromMatrixColumn(this.persp.matrix, 0)
    const up = new THREE.Vector3().setFromMatrixColumn(this.persp.matrix, 1)
    this.orbit.target.addScaledVector(right, -dx * wpp).addScaledVector(up, dy * wpp)
    this.updateCam()
  }

  private zoom(factor: number) {
    this.orbit.r = THREE.MathUtils.clamp(this.orbit.r * factor, 1, 1e5)
    this.updateCam()
  }

  // ---------- input ----------

  private bindInput() {
    const cv = this.renderer.domElement
    const opts = { signal: this.abort.signal }
    const btn = { L: false, M: false, R: false }
    const names = ['L', 'M', 'R'] as const
    let last: [number, number] | null = null
    let moved = 0
    let chordMoved = 0
    let zoomArmed = false
    let clickStart = false
    let draggingHandle = false

    cv.addEventListener('contextmenu', (e) => e.preventDefault(), opts)
    cv.addEventListener('mousedown', (e) => {
      const b = names[e.button]
      if (!b) return
      e.preventDefault()
      if (b !== 'M' && btn.M) chordMoved = 0
      btn[b] = true
      last = [e.clientX, e.clientY]
      if (b === 'L' && !btn.M) {
        draggingHandle = this.overHandle(e.clientX, e.clientY)
        clickStart = !draggingHandle
        moved = 0
      }
    }, opts)
    window.addEventListener('mouseup', (e) => {
      const b = names[e.button]
      if (!b) return
      if (b === 'L' && clickStart && !btn.M && moved < 4 && e.target === cv) this.select(this.pickAt(e.clientX, e.clientY))
      btn[b] = false
      if (b === 'M') zoomArmed = false
      else if (btn.M && this.mouseScheme === 'catia' && chordMoved < 4) zoomArmed = true
      if (b === 'L') {
        clickStart = false
        draggingHandle = false
      }
      if (!btn.L && !btn.M && !btn.R) last = null
    }, opts)
    window.addEventListener('mousemove', (e) => {
      if (!last) {
        if (e.target !== cv) return
        const onHandle = this.overHandle(e.clientX, e.clientY)
        const pick = onHandle ? null : this.pickAt(e.clientX, e.clientY)
        this.setHover(pick)
        cv.style.cursor = onHandle ? 'grab' : pick ? 'pointer' : ''
        return
      }
      const dx = e.clientX - last[0]
      const dy = e.clientY - last[1]
      last = [e.clientX, e.clientY]
      moved += Math.abs(dx) + Math.abs(dy)
      chordMoved += Math.abs(dx) + Math.abs(dy)
      if (draggingHandle) {
        cv.style.cursor = 'grabbing'
        this.handleDrag(dx, dy)
      } else if (this.mouseScheme === 'catia') {
        if (btn.M && (btn.L || btn.R)) this.rotate(dx, dy)
        else if (btn.M && zoomArmed) this.zoom(Math.exp(dy * 0.006))
        else if (btn.M) this.pan(dx, dy)
      } else if (btn.L) this.rotate(dx, dy)
      else if (btn.R || btn.M) this.pan(dx, dy)
    }, opts)
    cv.addEventListener('mouseleave', () => this.setHover(null), opts)
    cv.addEventListener('wheel', (e) => {
      e.preventDefault()
      this.zoom(Math.exp(e.deltaY * 0.0012))
    }, { ...opts, passive: false })
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.select(null)
    }, opts)
  }

  // ---------- render ----------

  private loop = () => {
    const w = this.el.clientWidth
    const h = this.el.clientHeight
    const r = this.renderer
    r.setViewport(0, 0, w, h)
    this.findingGroup.traverse((n) => {
      if (n.userData.billboard) n.quaternion.copy(this.cam.quaternion)
    })
    r.render(this.scene, this.cam)
    this.placePeakFlag(w, h)
    const s = 78
    r.setScissorTest(true)
    r.setScissor(w - s - 6, 6, s, s)
    r.setViewport(w - s - 6, 6, s, s)
    this.triCam.position.copy(this.persp.position).sub(this.orbit.target).setLength(4.4)
    this.triCam.lookAt(0, 0, 0)
    r.autoClear = false
    r.clearDepth()
    r.render(this.triScene, this.triCam)
    r.autoClear = true
    r.setScissorTest(false)
    this.frame = requestAnimationFrame(this.loop)
  }
}

function clearGroup(group: THREE.Group) {
  for (const child of [...group.children]) {
    group.remove(child)
    child.traverse((n) => {
      const m = n as THREE.Mesh
      m.geometry?.dispose()
      ;(m.material as THREE.Material | undefined)?.dispose()
    })
  }
}

function same(a: FacePick | null, b: FacePick | null) {
  return !!a && !!b && a.part === b.part && a.faceId === b.faceId && a.edgeId === b.edgeId
}

const NO_RAYCAST = () => {}

/** A colour-map strip for contours: 256 texels blended (a gradient), or `bands` flat colours with hard edges. */
export function colorStrip(bands: number): THREE.DataTexture {
  const n = bands > 0 ? bands : 256
  const data = new Uint8Array(4 * n)
  for (let i = 0; i < n; i++) {
    const [r, g, b] = bands > 0 ? bandColor(i, n) : turbo(i / (n - 1))
    data.set([Math.round(255 * r), Math.round(255 * g), Math.round(255 * b), 255], 4 * i)
  }
  const tex = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.magFilter = tex.minFilter = bands > 0 ? THREE.NearestFilter : THREE.LinearFilter
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping
  tex.generateMipmaps = false
  tex.needsUpdate = true
  return tex
}

/** Element edges as line segments sharing a mesh's position attribute (so a deformed shape moves them too). */
function edgeLines(position: THREE.BufferAttribute, pairs: number[], color: number, opacity: number): THREE.LineSegments {
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', position)
  geo.setIndex(pairs)
  const lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }))
  lines.renderOrder = 2
  lines.raycast = NO_RAYCAST
  return lines
}

/** Band `i` of `n` on the blue-to-red rainbow of banded stress plots (hue 240° down to 0°, full saturation). */
export function bandColor(i: number, n: number): [number, number, number] {
  const h = (240 * (1 - i / Math.max(1, n - 1))) / 60
  const x = 1 - Math.abs((h % 2) - 1)
  const k = Math.floor(h) % 6
  return ([[1, x, 0], [x, 1, 0], [0, 1, x], [0, x, 1], [x, 0, 1], [1, 0, x]] as [number, number, number][])[k]
}

/** Turbo colour map (Google, Apache 2.0), polynomial approximation: t in [0, 1] -> rgb in [0, 1]. */
export function turbo(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t))
  const r = 0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
  const b = 0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
  return [Math.min(1, Math.max(0, r)), Math.min(1, Math.max(0, g)), Math.min(1, Math.max(0, b))]
}
