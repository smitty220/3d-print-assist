export type Units = 'in' | 'mm'

const MM_PER_IN = 25.4

export function len(mm: number, u: Units, digits = u === 'in' ? 3 : 2) {
  return (u === 'in' ? mm / MM_PER_IN : mm).toFixed(digits)
}

export function area(mm2: number, u: Units) {
  return u === 'in' ? `${(mm2 / MM_PER_IN ** 2).toFixed(3)} in²` : `${mm2.toFixed(1)} mm²`
}

export function point(p: [number, number, number], u: Units) {
  return `(${p.map((v) => len(v, u)).join(', ')})`
}

export function dir(d: [number, number, number]) {
  const axes = ['X', 'Y', 'Z']
  const i = d.findIndex((v) => Math.abs(Math.abs(v) - 1) < 1e-6)
  if (i >= 0) return `${d[i] > 0 ? '+' : '−'}${axes[i]}`
  return `(${d.map((v) => v.toFixed(3)).join(', ')})`
}

export type Quantity = 'length' | 'force' | 'moment' | 'mass' | 'ratio' | 'stress'

const FACTOR: Record<Units, Record<Quantity, number>> = {
  // display value = internal SI value / factor
  in: { length: MM_PER_IN, force: 4.4482216, moment: 112.98483, mass: 0.45359237, ratio: 1, stress: 6.894757 },
  mm: { length: 1, force: 1, moment: 1, mass: 1, ratio: 1, stress: 1 },
}

export const UNIT_LABEL: Record<Units, Record<Quantity, string>> = {
  in: { length: 'in', force: 'lbf', moment: 'in·lbf', mass: 'lb', ratio: '', stress: 'ksi' },
  mm: { length: 'mm', force: 'N', moment: 'N·mm', mass: 'kg', ratio: '', stress: 'MPa' },
}

export const toDisplay = (v: number, q: Quantity, u: Units) => v / FACTOR[u][q]
export const fromDisplay = (v: number, q: Quantity, u: Units) => v * FACTOR[u][q]
