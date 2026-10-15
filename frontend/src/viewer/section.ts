import * as THREE from 'three'
import { inPlaneAxes } from './sectionPlane'

/**
 * Section tool: one clipping plane, hatched caps on every cut solid, and an on-screen plane gizmo.
 *
 * Caps use the stencil technique: for each part, back faces increment and front faces decrement the
 * stencil buffer (both clipped by the plane), so pixels where the plane passes through solid material
 * end up non-zero. A large quad on the plane is then drawn only where the stencil is non-zero.
 *
 * Convention (drafting): the plane normal points in the direction of sight, toward the material kept.
 */

export interface CapStyle {
  fill: number
  line: number
  /** Hatch angle in degrees, screen space. Adjacent parts alternate 45° / 135°. */
  angle: number
}

// Hatch is laid out in the section plane's own U/V coordinates (model space), so it stays attached to
// the cut face while the view rotates. Line width is held constant in screen pixels via fwidth().
const HATCH_VERT = /* glsl */ `
varying vec3 vWorld;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}`

const HATCH_FRAG = /* glsl */ `
uniform vec3 fillColor;
uniform vec3 lineColor;
uniform vec3 origin;
uniform vec3 axisU;
uniform vec3 axisV;
uniform float spacing;
uniform float widthPx;
uniform float angle;
varying vec3 vWorld;
void main() {
  vec3 r = vWorld - origin;
  vec2 p = vec2(dot(r, axisU), dot(r, axisV));
  float s = dot(p, vec2(-sin(angle), cos(angle))) / spacing;
  float d = abs(fract(s) - 0.5);
  float px = fwidth(s);
  float line = 1.0 - smoothstep(0.5 * widthPx * px, (0.5 * widthPx + 1.0) * px, d);
  gl_FragColor = vec4(mix(fillColor, lineColor, line), 1.0);
  #include <colorspace_fragment>
}`

interface PartSection {
  stencil: THREE.Group
  cap: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
}

export class SectionTool {
  readonly plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
  /** Stencil writers and caps, added to the main scene. */
  readonly caps = new THREE.Group()
  /** Plane outline and drag handle. */
  readonly gizmo = new THREE.Group()
  readonly gizmoCenter = new THREE.Vector3()
  handleLength = 10

  private parts = new Map<string, PartSection>()
  private handle = new THREE.Group()
  private planeFrame = new THREE.Group()
  private pixelRatio: number

  constructor(pixelRatio: number) {
    this.pixelRatio = pixelRatio
    this.caps.visible = false
    this.gizmo.visible = false

    const square = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, -0.5, 0), new THREE.Vector3(0.5, -0.5, 0),
      new THREE.Vector3(0.5, 0.5, 0), new THREE.Vector3(-0.5, 0.5, 0),
    ])
    const outline = new THREE.LineLoop(square, new THREE.LineBasicMaterial({ color: 0xe8870e, transparent: true, opacity: 0.9 }))
    const tint = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ color: 0xe8870e, transparent: true, opacity: 0.06, depthWrite: false, side: THREE.DoubleSide }),
    )
    outline.raycast = () => {}
    tint.raycast = () => {}
    this.planeFrame.add(outline, tint)

    const handleMat = new THREE.MeshBasicMaterial({ color: 0xff9a1f, depthTest: false, transparent: true })
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.75, 12).rotateX(Math.PI / 2).translate(0, 0, 0.375), handleMat)
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.25, 20).rotateX(Math.PI / 2).translate(0, 0, 0.875), handleMat)
    const grip = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), handleMat)
    for (const m of [shaft, cone, grip]) m.renderOrder = 1000
    this.handle.add(shaft, cone, grip)
    this.gizmo.add(this.planeFrame, this.handle)
  }

  addPart(name: string, geometry: THREE.BufferGeometry, style: CapStyle) {
    this.removePart(name)
    const writer = (side: THREE.Side, op: THREE.StencilOp) => {
      const m = new THREE.MeshBasicMaterial({
        side,
        colorWrite: false,
        depthWrite: false,
        depthTest: false,
        stencilWrite: true,
        stencilFunc: THREE.AlwaysStencilFunc,
        stencilFail: op,
        stencilZFail: op,
        stencilZPass: op,
      })
      m.clippingPlanes = [this.plane]
      return new THREE.Mesh(geometry, m)
    }
    const stencil = new THREE.Group()
    stencil.add(writer(THREE.BackSide, THREE.IncrementWrapStencilOp), writer(THREE.FrontSide, THREE.DecrementWrapStencilOp))
    stencil.traverse((o) => (o.raycast = () => {}))

    const capMat = new THREE.ShaderMaterial({
      vertexShader: HATCH_VERT,
      fragmentShader: HATCH_FRAG,
      uniforms: {
        fillColor: { value: new THREE.Color(style.fill) },
        lineColor: { value: new THREE.Color(style.line) },
        origin: { value: new THREE.Vector3() },
        axisU: { value: new THREE.Vector3(1, 0, 0) },
        axisV: { value: new THREE.Vector3(0, 1, 0) },
        spacing: { value: 2 },
        widthPx: { value: 1.3 * this.pixelRatio },
        angle: { value: THREE.MathUtils.degToRad(style.angle) },
      },
      side: THREE.DoubleSide,
      stencilWrite: true,
      stencilRef: 0,
      stencilFunc: THREE.NotEqualStencilFunc,
      stencilFail: THREE.ReplaceStencilOp,
      stencilZFail: THREE.ReplaceStencilOp,
      stencilZPass: THREE.ReplaceStencilOp,
    })
    const cap = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), capMat)
    cap.onAfterRender = (renderer) => renderer.clearStencil()
    cap.raycast = () => {}
    cap.userData.part = name

    this.caps.add(stencil, cap)
    this.parts.set(name, { stencil, cap })
    this.reorder()
  }

  removePart(name: string) {
    const p = this.parts.get(name)
    if (!p) return
    this.caps.remove(p.stencil, p.cap)
    p.stencil.traverse((o) => ((o as THREE.Mesh).material as THREE.Material | undefined)?.dispose())
    p.cap.geometry.dispose()
    p.cap.material.dispose()
    this.parts.delete(name)
    this.reorder()
  }

  /** Each part's stencil pass and cap must draw back-to-back, after all solid geometry. */
  private reorder() {
    let i = 0
    for (const p of this.parts.values()) {
      p.stencil.children.forEach((c) => (c.renderOrder = 10 + 2 * i))
      p.cap.renderOrder = 11 + 2 * i
      i++
    }
  }

  setPlane(normal: THREE.Vector3, point: THREE.Vector3, sceneCenter: THREE.Vector3, sceneRadius: number) {
    const n = normal.clone().normalize()
    this.plane.setFromNormalAndCoplanarPoint(n, point)
    this.plane.projectPoint(sceneCenter, this.gizmoCenter)
    // Orient the outline with the plane's own U/V axes so it reads square-on when viewed normal to it.
    const [u, v] = inPlaneAxes(n)
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(u, v, n))

    this.gizmo.position.copy(this.gizmoCenter)
    this.gizmo.quaternion.copy(q)
    this.planeFrame.scale.setScalar(sceneRadius * 2.2)
    this.handleLength = sceneRadius * 0.45
    this.handle.scale.setScalar(this.handleLength)

    // gizmoCenter only moves along the normal as the offset changes, so using it as the hatch origin
    // keeps the pattern fixed in the plane while the plane slides.
    const spacing = sceneRadius / 22
    for (const p of this.parts.values()) {
      p.cap.position.copy(this.gizmoCenter)
      p.cap.quaternion.copy(q)
      p.cap.scale.setScalar(sceneRadius * 6)
      const un = p.cap.material.uniforms
      un.origin.value.copy(this.gizmoCenter)
      un.axisU.value.copy(u)
      un.axisV.value.copy(v)
      un.spacing.value = spacing
    }
  }

  setPartVisible(name: string, visible: boolean) {
    const p = this.parts.get(name)
    if (!p) return
    p.stencil.visible = visible
    p.cap.visible = visible
  }

  setEnabled(on: boolean) {
    this.caps.visible = on
    this.gizmo.visible = on
  }

  setShowPlane(show: boolean) {
    this.planeFrame.visible = show
  }

  hitsHandle(raycaster: THREE.Raycaster) {
    return this.gizmo.visible && raycaster.intersectObject(this.handle, true).length > 0
  }
}
