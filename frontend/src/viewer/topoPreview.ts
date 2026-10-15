import * as THREE from 'three'
import type { TopoPreview } from '../api'

const SOLID = 0x2f9e8f
const FIXED = 0xe0782a
const NORMAL = 0xd9b52a
const LOAD = 0x8a4fd6

/** The topology set-up: forced-solid elements as cubes, supported nodes as dots, one arrow per load case. */
export function buildTopoPreview(p: TopoPreview, arrowLength: number): THREE.Group {
  const g = new THREE.Group()
  const n = p.solid.length / 3
  if (n) {
    const s = p.h * 0.92
    const cubes = new THREE.InstancedMesh(
      new THREE.BoxGeometry(s, s, s),
      new THREE.MeshStandardMaterial({ color: SOLID, transparent: true, opacity: 0.55, roughness: 0.6, depthWrite: false }),
      n,
    )
    const ax = p.axes
    const rot = new THREE.Matrix4().makeBasis(new THREE.Vector3(...ax[0]), new THREE.Vector3(...ax[1]), new THREE.Vector3(...ax[2]))
    const m = new THREE.Matrix4()
    for (let i = 0; i < n; i++) {
      m.copy(rot).setPosition(p.solid[3 * i], p.solid[3 * i + 1], p.solid[3 * i + 2])
      cubes.setMatrixAt(i, m)
    }
    cubes.renderOrder = 2
    g.add(cubes)
  }
  for (const s of p.supports) g.add(dots(s.points, s.kind === 'fixed' || s.kind === 'pin' ? FIXED : NORMAL, p.h * 0.7))
  for (const l of p.loads) {
    g.add(dots(l.points, LOAD, p.h * 0.55))
    const k = l.points.length / 3
    const c = new THREE.Vector3()
    const f = new THREE.Vector3()
    for (let i = 0; i < k; i++) {
      c.x += l.points[3 * i] / k
      c.y += l.points[3 * i + 1] / k
      c.z += l.points[3 * i + 2] / k
      f.x += l.forces[3 * i]
      f.y += l.forces[3 * i + 1]
      f.z += l.forces[3 * i + 2]
    }
    if (f.lengthSq() > 0) {
      const d = f.clone().normalize()
      // The arrow ends at the load's centre, pointing the way the load acts.
      const arrow = new THREE.ArrowHelper(d, c.clone().addScaledVector(d, -arrowLength), arrowLength, LOAD, arrowLength * 0.22, arrowLength * 0.1)
      arrow.traverse((o) => ((o as THREE.Mesh).renderOrder = 3))
      g.add(arrow)
    }
  }
  g.traverse((o) => (o.raycast = () => {}))
  return g
}

function dots(points: number[], color: number, size: number) {
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color, size, sizeAttenuation: true, depthTest: false }))
  pts.renderOrder = 4
  return pts
}
