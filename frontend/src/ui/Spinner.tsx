import type { CSSProperties } from 'react'

interface Props {
  size?: number
  thickness?: number
  label?: string
  decorative?: boolean
  className?: string
}

/** Accessible indeterminate loading spinner built on native `<progress>`. */
export function Spinner({ size = 16, thickness = 2, label = 'Loading', decorative = false, className = '' }: Props) {
  const style = {
    '--size': `${size}px`,
    '--thickness': `${thickness}px`,
  } as CSSProperties
  return (
    <progress
      className={`loading-spinner${className ? ` ${className}` : ''}`}
      style={style}
      aria-label={decorative ? undefined : label}
      aria-hidden={decorative ? true : undefined}
    />
  )
}
