import { useCallback, useEffect, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'

export interface PanelBox {
  x: number
  y: number
  w: number
  h: number
}

interface Options {
  storageKey: string
  side: 'left' | 'right'
  defaultWidth: number
  minWidth?: number
  maxWidth?: number
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

function clampBox(b: PanelBox, minW: number): PanelBox {
  const vw = typeof window !== 'undefined' ? window.innerWidth : 1280
  const vh = typeof window !== 'undefined' ? window.innerHeight : 800
  const w = clamp(b.w, minW, Math.max(minW, vw - 24))
  const h = clamp(b.h, 200, Math.max(200, vh - 24))
  const x = clamp(b.x, 8, Math.max(8, vw - w - 8))
  const y = clamp(b.y, 8, Math.max(8, vh - h - 8))
  return { x, y, w, h }
}

function readNum(key: string, fallback: number): number {
  try {
    const v = Number(localStorage.getItem(key))
    return Number.isFinite(v) && v > 0 ? v : fallback
  } catch {
    return fallback
  }
}

function readBool(key: string): boolean {
  try {
    return localStorage.getItem(key) === '1'
  } catch {
    return false
  }
}

function readBox(key: string, side: 'left' | 'right', defaultWidth: number, minW: number): PanelBox {
  const vw = typeof window !== 'undefined' ? window.innerWidth : 1280
  const vh = typeof window !== 'undefined' ? window.innerHeight : 800
  const fallback: PanelBox = {
    x: side === 'left' ? 16 : Math.max(16, vw - defaultWidth - 16),
    y: 64,
    w: defaultWidth,
    h: Math.min(560, Math.max(260, vh - 120)),
  }
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return clampBox(fallback, minW)
    const parsed = JSON.parse(raw) as Partial<PanelBox>
    return clampBox({
      x: typeof parsed.x === 'number' ? parsed.x : fallback.x,
      y: typeof parsed.y === 'number' ? parsed.y : fallback.y,
      w: typeof parsed.w === 'number' ? parsed.w : fallback.w,
      h: typeof parsed.h === 'number' ? parsed.h : fallback.h,
    }, minW)
  } catch {
    return clampBox(fallback, minW)
  }
}

export function usePanelLayout({ storageKey, side, defaultWidth, minWidth = 180, maxWidth = 620 }: Options) {
  const [collapsed, setCollapsed] = useState(() => readBool(`${storageKey}.collapsed`))
  const [floating, setFloating] = useState(() => readBool(`${storageKey}.floating`))
  const [width, setWidth] = useState(() => clamp(readNum(`${storageKey}.width`, defaultWidth), minWidth, maxWidth))
  const [box, setBox] = useState<PanelBox>(() => readBox(`${storageKey}.box`, side, defaultWidth, minWidth))
  const [dragging, setDragging] = useState(false)

  useEffect(() => { localStorage.setItem(`${storageKey}.collapsed`, collapsed ? '1' : '0') }, [storageKey, collapsed])
  useEffect(() => { localStorage.setItem(`${storageKey}.floating`, floating ? '1' : '0') }, [storageKey, floating])
  useEffect(() => { localStorage.setItem(`${storageKey}.width`, String(Math.round(width))) }, [storageKey, width])
  useEffect(() => { localStorage.setItem(`${storageKey}.box`, JSON.stringify(box)) }, [storageKey, box])

  useEffect(() => {
    const onResize = () => setBox((prev) => clampBox(prev, minWidth))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [minWidth])

  // Track drag origins in refs to avoid stale closures during pointermove.
  const resizeOrigin = useRef<{ startX: number; startW: number } | null>(null)
  const moveOrigin = useRef<{ startX: number; startY: number; box: PanelBox; moved: boolean } | null>(null)
  const floatResizeOrigin = useRef<{ startX: number; startY: number; startW: number; startH: number } | null>(null)

  const onDockResizeDown = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    resizeOrigin.current = { startX: e.clientX, startW: width }
    e.currentTarget.setPointerCapture(e.pointerId)
    setDragging(true)
  }, [width])

  const onDockResizeMove = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    const o = resizeOrigin.current
    if (!o || !e.currentTarget.hasPointerCapture(e.pointerId)) return
    const dx = e.clientX - o.startX
    const next = side === 'left' ? o.startW + dx : o.startW - dx
    setWidth(clamp(next, minWidth, maxWidth))
  }, [side, minWidth, maxWidth])

  const onDockResizeEnd = useCallback(() => {
    resizeOrigin.current = null
    setDragging(false)
  }, [])

  const onDockResizeKey = useCallback((e: ReactKeyboardEvent<HTMLElement>) => {
    const step = e.shiftKey ? 32 : 12
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      setWidth((w) => clamp(side === 'left' ? w - step : w + step, minWidth, maxWidth))
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      setWidth((w) => clamp(side === 'left' ? w + step : w - step, minWidth, maxWidth))
    }
  }, [side, minWidth, maxWidth])

  const onMoveDown = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    const aside = e.currentTarget.closest('aside')
    const rect = aside?.getBoundingClientRect()
    const baseBox = floating
      ? box
      : clampBox({
          x: rect ? rect.left : box.x,
          y: rect ? rect.top : 64,
          w: rect ? Math.max(minWidth, rect.width) : width,
          h: rect ? Math.max(240, rect.height) : box.h,
        }, minWidth)
    moveOrigin.current = { startX: e.clientX, startY: e.clientY, box: baseBox, moved: false }
    e.currentTarget.setPointerCapture(e.pointerId)
  }, [floating, box, width, minWidth])

  const onMoveMove = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    const o = moveOrigin.current
    if (!o || !e.currentTarget.hasPointerCapture(e.pointerId)) return
    const dx = e.clientX - o.startX
    const dy = e.clientY - o.startY
    if (!o.moved && Math.hypot(dx, dy) < 4) return
    o.moved = true
    setDragging(true)
    if (!floating) {
      setFloating(true)
      setCollapsed(false)
    }
    setBox(clampBox({ ...o.box, x: o.box.x + dx, y: o.box.y + dy }, minWidth))
  }, [floating, minWidth])

  const onMoveEnd = useCallback(() => {
    moveOrigin.current = null
    setDragging(false)
  }, [])

  const onMoveKey = useCallback((e: ReactKeyboardEvent<HTMLElement>) => {
    if (!floating) return
    const step = e.shiftKey ? 32 : 16
    if (e.key === 'ArrowLeft') { e.preventDefault(); setBox((b) => clampBox({ ...b, x: b.x - step }, minWidth)) }
    else if (e.key === 'ArrowRight') { e.preventDefault(); setBox((b) => clampBox({ ...b, x: b.x + step }, minWidth)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setBox((b) => clampBox({ ...b, y: b.y - step }, minWidth)) }
    else if (e.key === 'ArrowDown') { e.preventDefault(); setBox((b) => clampBox({ ...b, y: b.y + step }, minWidth)) }
  }, [floating, minWidth])

  const onFloatResizeDown = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    floatResizeOrigin.current = { startX: e.clientX, startY: e.clientY, startW: box.w, startH: box.h }
    e.currentTarget.setPointerCapture(e.pointerId)
    setDragging(true)
  }, [box.w, box.h])

  const onFloatResizeMove = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    const o = floatResizeOrigin.current
    if (!o || !e.currentTarget.hasPointerCapture(e.pointerId)) return
    const nextW = o.startW + (e.clientX - o.startX)
    const nextH = o.startH + (e.clientY - o.startY)
    setBox((b) => clampBox({ ...b, w: nextW, h: nextH }, minWidth))
  }, [minWidth])

  const onFloatResizeEnd = useCallback(() => {
    floatResizeOrigin.current = null
    setDragging(false)
  }, [])

  return {
    collapsed,
    setCollapsed,
    floating,
    setFloating,
    width,
    resetWidth: () => setWidth(defaultWidth),
    box,
    dragging,
    dockResizeProps: {
      onPointerDown: onDockResizeDown,
      onPointerMove: onDockResizeMove,
      onLostPointerCapture: onDockResizeEnd,
      onKeyDown: onDockResizeKey,
      onDoubleClick: () => setWidth(defaultWidth),
    },
    moveHandleProps: {
      onPointerDown: onMoveDown,
      onPointerMove: onMoveMove,
      onLostPointerCapture: onMoveEnd,
      onKeyDown: onMoveKey,
    },
    floatResizeProps: {
      onPointerDown: onFloatResizeDown,
      onPointerMove: onFloatResizeMove,
      onLostPointerCapture: onFloatResizeEnd,
    },
  }
}
