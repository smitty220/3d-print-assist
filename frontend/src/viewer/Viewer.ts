import * as THREE from 'three'
import type { ShapeMesh } from '../api'
import { SectionTool, type CapStyle } from './section'

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

/** Three.js viewport: left-drag orbits, right- or middle-drag pans, the wheel zooms toward the cursor. Per-B-rep-face
 *  and edge picking, and a section tool. Scene units are mm, Z up (the print bed is Z = 0). */
export class Viewer {
  onSelect?: (pick: FacePick | null) => void
  /** Called while the section handle is dragged, with the plane's movement along its normal in mm. */
  onSectionDrag?: (deltaMm: number) => void
  /** What a click picks: B-rep faces (default) or edges (for blends). */
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
  private edgeHoverGroup = new THREE.Group()
  private edgeMarkGroup = new THREE.Group()
  private abort = new AbortController()
  private frame = 0
  private resizeObs: ResizeObserver

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
    this.scene.add(this.section.caps, this.section.gizmo, this.edgeHoverGroup, this.edgeMarkGroup)

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

  // ---------- overlays ----------

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

  // ---------- section ----------

  setSection(on: boolean) {
    this.sectionOn = on
    this.section.setEnabled(on)
    for (const p of this.parts.values()) this.applyClipping(p)
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

  /** Zoom toward the point under the cursor: the target slides toward it as the camera closes in. */
  private zoomAt(factor: number, clientX: number, clientY: number) {
    const r = this.renderer.domElement.getBoundingClientRect()
    const before = this.orbit.r
    const next = THREE.MathUtils.clamp(before * factor, 1, 1e5)
    const wpp = (2 * this.halfHeight()) / (r.height || 1)
    const dx = (clientX - r.left - r.width / 2) * wpp
    const dy = (clientY - r.top - r.height / 2) * wpp
    const right = new THREE.Vector3().setFromMatrixColumn(this.persp.matrix, 0)
    const up = new THREE.Vector3().setFromMatrixColumn(this.persp.matrix, 1)
    const k = 1 - next / before
    this.orbit.target.addScaledVector(right, dx * k).addScaledVector(up, -dy * k)
    this.orbit.r = next
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
    let clickStart = false
    let draggingHandle = false

    cv.addEventListener('contextmenu', (e) => e.preventDefault(), opts)
    cv.addEventListener('mousedown', (e) => {
      const b = names[e.button]
      if (!b) return
      e.preventDefault()
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
      if (draggingHandle) {
        cv.style.cursor = 'grabbing'
        this.handleDrag(dx, dy)
      } else if (btn.L) this.rotate(dx, dy)
      else if (btn.R || btn.M) this.pan(dx, dy)
    }, opts)
    cv.addEventListener('mouseleave', () => this.setHover(null), opts)
    cv.addEventListener('wheel', (e) => {
      e.preventDefault()
      this.zoomAt(Math.exp(e.deltaY * 0.0012), e.clientX, e.clientY)
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
    r.render(this.scene, this.cam)
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
