import { Icon } from '../ui/icons'
import { PART_TOOLS, type Tool } from './tools'

export function PartToolbar({ onTool, active }: { onTool: (t: Tool) => void; active: string | null }) {
  return (
    <div className="toolbar" role="toolbar" aria-label="Part tools">
      {PART_TOOLS.map((g) => (
        <div key={g.group} className="tgroup">
          <div className="tbtns">
            {g.tools.map((t) => (
              <button key={t.id} className="tool" disabled={!!t.later} aria-pressed={active === t.id}
                title={t.later ? `${t.label} (coming in ${t.later})` : t.label} onClick={() => onTool(t)}>
                <Icon name={t.icon} size={21} color={t.later ? 'var(--muted)' : t.color ?? 'var(--fg)'} width={1.6} />
                <span>{t.label}</span>
              </button>
            ))}
          </div>
          <div className="tname">{g.group}</div>
        </div>
      ))}
    </div>
  )
}
