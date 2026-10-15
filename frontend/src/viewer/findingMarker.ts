import * as THREE from 'three'
import type { Finding } from '../api'

const COLOR = { ok: 0x2e9e5b, warn: 0xe08a12, fail: 0xd8433a } as const

function onTop<T extends THREE.Object3D>(o: T): T {
  // Findings usually sit inside or behind geometry (a clearance between two parts); draw them over everything.
  o.traverse((n) => {
    const m = (n as THREE.Mesh).material as THREE.Material | undefined
    if (m) {
      m.depthTest = false
      m.transparent = true
    }
    n.renderOrder = 950
    n.raycast = () => {}
  })
  return o
}

/** One point: a ringed dot. Two points: a dimension line between them with end dots (clearance, wall thickness). */
export function buildFindingMarker(f: Finding, size: number): THREE.Group | null {
  if (!f.points.length) return null
  const color = COLOR[f.status]
  const g = new THREE.Group()
  const dot = new THREE.SphereGeometry(size * 0.55, 18, 12)
  const fill = new THREE.MeshBasicMaterial({ color })
  const pts = f.points.map((p) => new THREE.Vector3(...p))
  for (const p of pts) {
    const m = new THREE.Mesh(dot, fill)
    m.position.copy(p)
    g.add(m)
  }
  if (pts.length >= 2 && pts[0].distanceTo(pts[1]) > 1e-6) {
    // WebGL ignores line widths, so the dimension is a thin rod.
    const [a, b] = pts
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.16, size * 0.16, a.distanceTo(b), 10), fill)
    rod.position.copy(a).add(b).multiplyScalar(0.5)
    rod.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize())
    g.add(rod)
    const ring = new THREE.Mesh(new THREE.RingGeometry(size * 1.3, size * 1.6, 40), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, opacity: 0.9 }))
    ring.position.copy(rod.position)
    ring.userData.billboard = true
    g.add(ring)
  } else {
    // A camera-facing ring around the single point.
    const ring = new THREE.Mesh(new THREE.RingGeometry(size * 0.9, size * 1.15, 40), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, opacity: 0.9 }))
    ring.position.copy(pts[0])
    ring.userData.billboard = true
    g.add(ring)
  }
  return onTop(g)
}
