import * as THREE from 'three'

export type Vec3 = [number, number, number]
export type SectionBase = 'YZ' | 'ZX' | 'XY' | 'face'

/**
 * A section plane as the user defines it: a base orientation, a reference point, two tilts about the
 * base plane's in-plane axes, and an offset along the resulting normal. The normal points in the
 * direction of sight, toward the material that is kept.
 */
export interface SectionSpec {
  base: SectionBase
  baseNormal: Vec3
  origin: Vec3
  tiltA: number // degrees, about in-plane axis U
  tiltB: number // degrees, about in-plane axis V
  offset: number // mm along the final normal
  flip: boolean
  source?: string
}

export const BASE_NORMALS: Record<Exclude<SectionBase, 'face'>, Vec3> = {
  YZ: [1, 0, 0],
  ZX: [0, 1, 0],
  XY: [0, 0, 1],
}

/** Two unit vectors spanning the plane with normal n, chosen deterministically (U is horizontal when possible). */
export function inPlaneAxes(n: THREE.Vector3): [THREE.Vector3, THREE.Vector3] {
  const helper = Math.abs(n.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0)
  const u = new THREE.Vector3().crossVectors(helper, n).normalize()
  const v = new THREE.Vector3().crossVectors(n, u).normalize()
  return [u, v]
}

export function resolveSection(s: SectionSpec): { normal: THREE.Vector3; point: THREE.Vector3 } {
  const n0 = new THREE.Vector3(...s.baseNormal).normalize()
  if (s.flip) n0.negate()
  const [u, v] = inPlaneAxes(n0)
  const normal = n0
    .clone()
    .applyAxisAngle(u, THREE.MathUtils.degToRad(s.tiltA))
    .applyAxisAngle(v, THREE.MathUtils.degToRad(s.tiltB))
    .normalize()
  const point = new THREE.Vector3(...s.origin).addScaledVector(normal, s.offset)
  return { normal, point }
}

/** Plane through a cylinder's axis, facing the viewer as squarely as possible. */
export function normalThroughAxis(axisDir: Vec3, viewDir: THREE.Vector3): Vec3 {
  const a = new THREE.Vector3(...axisDir).normalize()
  const n = viewDir.clone().addScaledVector(a, -viewDir.dot(a))
  if (n.lengthSq() < 1e-8) n.copy(inPlaneAxes(a)[0])
  n.normalize()
  return [n.x, n.y, n.z]
}
