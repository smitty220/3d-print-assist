// Stroke icons (24 × 24) for the toolbars and the tree.
export const ICONS: Record<string, string> = {
  eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM9 12a3 3 0 1 0 6 0a3 3 0 1 0-6 0',
  sketch: 'M4 20h16M4 20V4M8 16l8-8M14 6l4 4', plane: 'M3 15l6-8h12l-6 8z',
  extrude: 'M5 18h10v-6H5zM5 12l4-5h10l-4 5M15 18l4-5V7', revolve: 'M12 2v20M7 7a5 2 0 0 0 10 0a5 2 0 0 0-10 0M7 7v10a5 2 0 0 0 10 0V7',
  sweep: 'M4 19c5 0 4-14 9-14h7M2 16h4v6H2z', loft: 'M6 19h12M9 5h6M6 19L9 5M18 19L15 5', rib: 'M4 18h16M4 18V7M20 18V7M12 18V9',
  text: 'M4 6V4h16v2M12 4v16M9 20h6', emboss: 'M3 18h18M6 18v-8h5v8M14 18v-4h4v4', hole: 'M6 12a6 6 0 1 0 12 0a6 6 0 1 0-12 0M10 12a2 2 0 1 0 4 0a2 2 0 1 0-4 0',
  thread: 'M8 3v18M16 3v18M8 5l8 2M8 9l8 2M8 13l8 2M8 17l8 2', primitive: 'M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5',
  fillet: 'M4 20V11a7 7 0 0 1 7-7h9', chamfer: 'M4 20V10l6-6h10', shell: 'M4 5h16v15H4zM7.5 5v11.5h9V5', draft: 'M7 20l2-16h6l2 16z',
  press: 'M12 4v11M8 8l4-4 4 4M4 19h16', split: 'M4 4h16v16H4zM3 14l18-4', combine: 'M4 9a5 5 0 1 0 10 0a5 5 0 1 0-10 0M10 15a5 5 0 1 0 10 0a5 5 0 1 0-10 0',
  move: 'M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3',
  linear: 'M4 8h4v4H4zM10 8h4v4h-4zM16 8h4v4h-4zM4 14h4v4H4zM10 14h4v4h-4zM16 14h4v4h-4z',
  circular: 'M10 5a2 2 0 1 0 4 0a2 2 0 1 0-4 0M17 12a2 2 0 1 0 4 0a2 2 0 1 0-4 0M10 19a2 2 0 1 0 4 0a2 2 0 1 0-4 0M3 12a2 2 0 1 0 4 0a2 2 0 1 0-4 0',
  path: 'M3 19c5-11 13-11 18 0M4.5 13h3v3h-3zM10.5 8h3v3h-3zM16.5 13h3v3h-3z', mirror: 'M12 3v18M9 7L4 17h5zM15 7l5 10h-5z',
  measure: 'M3 16L16 3l5 5L8 21zM7 12l2 2M10 9l2 2M13 6l2 2', section: 'M4 6h16v12H4zM4 18L20 6', check: 'M5 12l4 4 10-10M4 20h16',
  vars: 'M8 4c-2 0-2 2-2 4s-1 4-2 4c1 0 2 2 2 4s0 4 2 4M16 4c2 0 2 2 2 4s1 4 2 4c-1 0-2 2-2 4s0 4-2 4M10 9l4 6M14 9l-4 6',
  color: 'M12 3c4 5 6 8 6 11a6 6 0 0 1-12 0c0-3 2-6 6-11z', importF: 'M12 3v12M7 10l5 5 5-5M4 20h16',
  part: 'M12 3l8 4.5v9L12 21l-8-4.5v-9z', folder: 'M3 7h7l2 2h9v10H3z', body: 'M5 8l7-4 7 4v8l-7 4-7-4z',
  param: 'M9 6L7 18M17 6l-2 12M5 10h14M4 14h14', relation: 'M4 12h4l3-7 3 14 3-7h3',
  line: 'M5 19L19 5', rect: 'M4 6h16v12H4z', centerRect: 'M4 6h16v12H4zM12 10v4M10 12h4', circle: 'M4 12a8 8 0 1 0 16 0a8 8 0 1 0-16 0', arc: 'M4 17a8 8 0 0 1 16 0',
  polygon: 'M12 3l8 5v8l-8 5-8-5V8z', slot: 'M8 8h8a4 4 0 0 1 0 8H8a4 4 0 0 1 0-8z', point: 'M10 12a2 2 0 1 0 4 0a2 2 0 1 0-4 0M12 4v4M12 16v4M4 12h4M16 12h4',
  construction: 'M4 20l3-3M9 15l3-3M14 10l3-3M19 5l1-1', trimSplit: 'M3 12h7M14 12h7M12 8v8', dimension: 'M4 7v10M20 7v10M4 12h16M7 10l-3 2 3 2M17 10l3 2-3 2',
  undo: 'M9 14L4 9l5-5M4 9h11a5 5 0 0 1 0 10h-3', bolt: 'M13 2L4 14h7l-1 8 9-12h-7z', ok: 'M5 12l4 4 10-10', warn: 'M12 3L2 21h20zM12 10v5M12 18v.5',
  fail: 'M6 6l12 12M18 6L6 18', skip: 'M6 12h12', lock: 'M5 11h14v10H5zM8 11V7a4 4 0 0 1 8 0v4', send: 'M4 12h15M13 6l6 6-6 6',
  chevL: 'M15 6l-6 6 6 6', chevR: 'M9 6l6 6-6 6',
}

export function Icon({ name, size = 18, color = 'currentColor', fill = 'none', width = 1.7, label }:
  { name: string; size?: number; color?: string; fill?: string; width?: number; label?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color} strokeWidth={width} strokeLinecap="round"
      strokeLinejoin="round" role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      <path d={ICONS[name] ?? ICONS.part} />
    </svg>
  )
}

export const STATUS: Record<string, { icon: string; color: string; fill: string; label: string }> = {
  stale: { icon: 'bolt', color: 'var(--stale)', fill: 'var(--stale)', label: 'Needs update' },
  ok: { icon: 'ok', color: 'var(--ok)', fill: 'none', label: 'Up to date' },
  warn: { icon: 'warn', color: 'var(--warn)', fill: 'none', label: 'Built, a check fails' },
  fail: { icon: 'fail', color: 'var(--bad)', fill: 'none', label: 'Failed' },
  skipped: { icon: 'skip', color: 'var(--muted)', fill: 'none', label: 'Skipped' },
}

export function StatusIcon({ status }: { status: string }) {
  const s = STATUS[status] ?? STATUS.ok
  return <Icon name={s.icon} size={15} color={s.color} fill={s.fill} width={2} label={s.label} />
}
