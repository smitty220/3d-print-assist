import * as THREE from 'three'
import { isInterface, type BoltPattern, type Bore, type DesignSpace, type Fastener, type Frame, type IntentObject, type LoadCase, type PlanarMount } from '../api'

/** Assumed bracket base thickness for drawing hardware until a generated part exists (0.25 in). */
export const ASSUMED_BASE_MM = 6.35
const WASHER_T = 1.6 // 0.063 in
const NUT_H = 4.8 // ~0.19 in

const BLUE = 0x2f86e0
const AMBER = 0xff9a1f
const PURPLE = 0x8a4fd6
const LOAD_ARROW_MM = 45

export function frameMatrix(f: Frame) {
  const z = new THREE.Vector3(...f.z).normalize()
  const x = new THREE.Vector3(...f.x).normalize()
  const y = new THREE.Vector3().crossVectors(z, x)
  return new THREE.Matrix4().makeBasis(x, y, z).setPosition(...f.origin)
}

interface Mats {
  fill: THREE.MeshBasicMaterial
  line: THREE.LineBasicMaterial
  metal: THREE.MeshStandardMaterial
  dark: THREE.MeshBasicMaterial
}

function mats(selected: boolean): Mats {
  const c = selected ? AMBER : BLUE
  return {
    fill: new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: selected ? 0.45 : 0.32, depthWrite: false, side: THREE.DoubleSide }),
    line: new THREE.LineBasicMaterial({ color: c }),
    metal: new THREE.MeshStandardMaterial({ color: selected ? 0xc9a060 : 0x9b9480, metalness: 0.6, roughness: 0.35 }),
    dark: new THREE.MeshBasicMaterial({ color: 0x1b1d20 }),
  }
}

/** Cylinder along local +Z from z0 to z1. */
function zCyl(r: number, z0: number, z1: number, mat: THREE.Material, segments = 28) {
  const g = new THREE.CylinderGeometry(r, r, Math.abs(z1 - z0), segments).rotateX(Math.PI / 2).translate(0, 0, (z0 + z1) / 2)
  return new THREE.Mesh(g, mat)
}

function hexPrism(acrossFlats: number, z0: number, z1: number, mat: THREE.Material) {
  const m = zCyl(acrossFlats / 2 / Math.cos(Math.PI / 6), z0, z1, mat, 6)
  m.rotation.z = Math.PI / 6
  return m
}

function mount(o: PlanarMount, m: Mats) {
  const g = new THREE.Group()
  const shape = new THREE.Shape(o.footprint.map(([x, y]) => new THREE.Vector2(x, y)))
  const patch = new THREE.Mesh(new THREE.ShapeGeometry(shape), m.fill)
  patch.position.z = 0.05
  const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(o.footprint.map(([x, y]) => new THREE.Vector3(x, y, 0.05))), m.line)
  const size = Math.max(...o.footprint.map(([x, y]) => Math.hypot(x, y)))
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), size * 0.35, m.line.color.getHex(), size * 0.08, size * 0.04)
  g.add(patch, outline, arrow)
  return g
}

function boltPattern(o: BoltPattern, m: Mats, catalog: Map<string, Fastener>) {
  const g = new THREE.Group()
  const f = o.fastener ? catalog.get(o.fastener) : undefined
  for (const h of o.holes) {
    const hg = new THREE.Group()
    hg.position.set(h.center[0], h.center[1], 0)
    const depth = h.depth ?? 6.35
    hg.add(zCyl(h.diameter * 0.55, -depth - 2, ASSUMED_BASE_MM + 2, m.fill, 20))
    if (f) {
      const top = ASSUMED_BASE_MM
      hg.add(zCyl(f.washer_od / 2, top, top + WASHER_T, m.metal))
      const headZ0 = top + WASHER_T
      if (f.head === 'socket') {
        hg.add(zCyl(f.head_size / 2, headZ0, headZ0 + f.head_height, m.metal, 32))
        hg.add(hexPrism(f.shank_d * 0.7, headZ0 + f.head_height, headZ0 + f.head_height + 0.05, m.dark))
      } else {
        hg.add(hexPrism(f.head_size, headZ0, headZ0 + f.head_height, m.metal))
      }
      const nutZ0 = -depth - WASHER_T
      hg.add(zCyl(f.washer_od / 2, -depth - WASHER_T, -depth, m.metal))
      hg.add(hexPrism(f.nut_af, nutZ0 - NUT_H, nutZ0, m.metal))
      hg.add(zCyl(f.shank_d / 2, nutZ0 - NUT_H - 1.5, headZ0, m.metal, 20))
    }
    g.add(hg)
  }
  return g
}

function bore(o: Bore, m: Mats) {
  const g = new THREE.Group()
  const half = o.length / 2
  g.add(zCyl(o.diameter / 2 + 0.05, -half, half, m.fill, 36))
  const ext = o.length * 0.5 + o.diameter
  const axis = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, -ext), new THREE.Vector3(0, 0, ext)]), new THREE.LineDashedMaterial({ color: m.line.color, dashSize: 2, gapSize: 1.2 }))
  axis.computeLineDistances()
  g.add(axis)
  return g
}

function designSpace(o: DesignSpace, selected: boolean) {
  const [x0, y0, z0] = o.min
  const [x1, y1, z1] = o.max
  const box = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0).translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
  const lines = new THREE.LineSegments(
    new THREE.EdgesGeometry(box),
    new THREE.LineDashedMaterial({ color: selected ? AMBER : 0x5f6b78, dashSize: 2.5, gapSize: 1.6 }),
  )
  lines.computeLineDistances()
  box.dispose()
  const g = new THREE.Group()
  g.add(lines)
  return g
}

/** Arrow ending at the target interface's origin, pointing along the applied force. Positioned in world space. */
function loadArrow(o: LoadCase, all: IntentObject[], selected: boolean) {
  const target = all.find((t) => t.id === o.target)
  if (!target || !isInterface(target)) return null
  const at = new THREE.Vector3(...target.frame.origin)
  const f = new THREE.Vector3(...o.force)
  const color = selected ? AMBER : PURPLE
  const g = new THREE.Group()
  if (f.lengthSq() < 1e-12) {
    g.add(new THREE.Mesh(new THREE.SphereGeometry(2.2, 16, 12), new THREE.MeshBasicMaterial({ color })))
    g.position.copy(at)
    return g
  }
  const d = f.normalize()
  const arrow = new THREE.ArrowHelper(d, at.clone().addScaledVector(d, -LOAD_ARROW_MM), LOAD_ARROW_MM, color, 9, 4.5)
  // Loads often act inside mating geometry (a pin in a bore); draw them on top so they stay visible.
  arrow.traverse((n) => {
    const m = (n as THREE.Mesh).material as THREE.Material | undefined
    if (m) m.depthTest = false
    n.renderOrder = 900
  })
  g.add(arrow)
  return g
}

/** Viewport representation of one intent object. Keep-outs draw nothing extra (their part shows red). */
export function buildOverlay(o: IntentObject, all: IntentObject[], catalog: Map<string, Fastener>, selected: boolean): THREE.Group | null {
  const m = mats(selected)
  let g: THREE.Group | null
  switch (o.kind) {
    case 'planar_mount':
      g = mount(o, m)
      break
    case 'bolt_pattern':
      g = boltPattern(o, m, catalog)
      break
    case 'bore':
      g = bore(o, m)
      break
    case 'design_space':
      g = designSpace(o, selected)
      break
    case 'load_case':
      g = loadArrow(o, all, selected)
      break
    default:
      g = null
  }
  if (!g) return null
  if ('frame' in o) g.applyMatrix4(frameMatrix(o.frame))
  g.userData.intentId = o.id
  g.traverse((n) => (n.raycast = () => {}))
  return g
}
