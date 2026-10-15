// The part toolbar: every tool from the plan, grouped. `kind` is the feature it adds; tools without one open a
// dialog or a mode. `later` marks tools still to come (shown disabled with the milestone).
export interface Tool { id: string; label: string; icon: string; kind?: string; action?: string; later?: string; color?: string }

export const PART_TOOLS: { group: string; tools: Tool[] }[] = [
  { group: 'Sketch', tools: [
    { id: 'sketch', label: 'Sketch', icon: 'sketch', action: 'new-sketch', color: 'var(--sketch)' },
    { id: 'plane', label: 'Plane', icon: 'plane', later: 'M3', color: 'var(--sketch)' },
  ] },
  { group: 'Create', tools: [
    { id: 'extrude', label: 'Extrude', icon: 'extrude', kind: 'extrude' },
    { id: 'revolve', label: 'Revolve', icon: 'revolve', kind: 'revolve' },
    { id: 'sweep', label: 'Sweep', icon: 'sweep', later: 'M3' },
    { id: 'loft', label: 'Loft', icon: 'loft', later: 'M3' },
    { id: 'rib', label: 'Rib', icon: 'rib', later: 'M3' },
    { id: 'text', label: 'Text', icon: 'text', kind: 'text' },
    { id: 'emboss', label: 'Emboss', icon: 'emboss', later: 'M3' },
    { id: 'hole', label: 'Hole', icon: 'hole', later: 'M3' },
    { id: 'thread', label: 'Thread', icon: 'thread', later: 'M3' },
    { id: 'primitive', label: 'Primitive', icon: 'primitive', kind: 'plate' },
  ] },
  { group: 'Modify', tools: [
    { id: 'fillet', label: 'Fillet', icon: 'fillet', kind: 'fillet' },
    { id: 'chamfer', label: 'Chamfer', icon: 'chamfer', kind: 'chamfer' },
    { id: 'shell', label: 'Shell', icon: 'shell', kind: 'shell' },
    { id: 'draft', label: 'Draft', icon: 'draft', later: 'M3' },
    { id: 'press', label: 'Press/pull', icon: 'press', later: 'M3' },
    { id: 'split', label: 'Split', icon: 'split', kind: 'split' },
    { id: 'combine', label: 'Combine', icon: 'combine', kind: 'combine' },
    { id: 'move', label: 'Move', icon: 'move', later: 'M3' },
  ] },
  { group: 'Pattern', tools: [
    { id: 'linear', label: 'Linear', icon: 'linear', later: 'M3' },
    { id: 'circular', label: 'Circular', icon: 'circular', later: 'M3' },
    { id: 'path', label: 'On path', icon: 'path', later: 'M3' },
    { id: 'mirror', label: 'Mirror', icon: 'mirror', later: 'M3' },
  ] },
  { group: 'Inspect', tools: [
    { id: 'measure', label: 'Measure', icon: 'measure', later: 'M3', color: 'var(--ok)' },
    { id: 'section', label: 'Section', icon: 'section', action: 'section', color: 'var(--ok)' },
    { id: 'checks', label: 'Print check', icon: 'check', action: 'checks', color: 'var(--ok)' },
  ] },
  { group: 'Design', tools: [
    { id: 'parameter', label: 'Parameter', icon: 'vars', action: 'new-parameter', color: 'var(--accent)' },
    { id: 'color', label: 'Color', icon: 'color', action: 'color', color: 'var(--accent)' },
    { id: 'import', label: 'Import', icon: 'importF', action: 'import', color: 'var(--accent)' },
  ] },
]
