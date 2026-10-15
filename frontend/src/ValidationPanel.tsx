import type { Finding, ValidationReport } from './api'
import { area, len, toDisplay, UNIT_LABEL, type Units } from './units'

const ICON = { ok: '✔', warn: '⚠', fail: '✖' } as const
const CLASS = { ok: 'ok', warn: 'wn', fail: 'no' } as const

export const findingKey = (f: Finding) => `${f.stage}:${f.title}`

export function valueText(f: Finding, units: Units) {
  if (f.value == null) return ''
  switch (f.quantity) {
    case 'length': return `${len(f.value, units)} ${units}`
    case 'area': return area(f.value, units)
    case 'percent': return `${f.value.toFixed(0)}%`
    case 'count': return String(Math.round(f.value))
    case 'mass': return `${toDisplay(f.value, 'mass', units).toFixed(3)} ${UNIT_LABEL[units].mass}`
    default: return f.value.toFixed(2)
  }
}

/** Overall state as a tree/status dot class. */
export function reportClass(r: ValidationReport | null) {
  if (!r) return ''
  return r.counts.fail ? 'no' : r.counts.warn ? 'wn' : 'ok'
}

interface Props {
  report: ValidationReport | null
  loading: boolean
  selected: string | null
  units: Units
  onSelect: (f: Finding) => void
  onRef: (id: string) => void
}

export function ValidationPanel({ report, loading, selected, units, onSelect, onRef }: Props) {
  if (!report) return <p className="note">{loading ? 'Checking…' : 'No validation results yet.'}</p>
  const { ok, warn, fail } = report.counts
  return (
    <>
      <div className="ttl">
        <b>Validation</b>
        <span>
          <span className="no">{ICON.fail} {fail}</span> · <span className="wn">{ICON.warn} {warn}</span> · <span className="ok">{ICON.ok} {ok}</span>
          {loading && ' · checking…'}
        </span>
      </div>
      <Group title="Pre-flight · intent" note={report.preflight_ready ? 'The intent is complete enough to build from.' : 'Fix the failures before generating a part: no part can satisfy this intent.'}
        findings={report.preflight} selected={selected} units={units} onSelect={onSelect} onRef={onRef} />
      <Group title="Part checks" note={report.part.length ? 'Checked against the intent on every rebuild.' : 'Start a part to check it against the intent.'}
        findings={report.part} selected={selected} units={units} onSelect={onSelect} onRef={onRef} />
      <p className="note">Click a check to mark its location in the viewport. Minimum wall is sampled by rays across every face, so treat it as a screen, not a guarantee.</p>
    </>
  )
}

function Group({ title, note, findings, selected, units, onSelect, onRef }: { title: string; note: string; findings: Finding[]; selected: string | null; units: Units; onSelect: (f: Finding) => void; onRef: (id: string) => void }) {
  // Problems first, then passes, keeping the backend's order within each.
  const rank = { fail: 0, warn: 1, ok: 2 } as const
  const sorted = [...findings].sort((a, b) => rank[a.status] - rank[b.status])
  return (
    <>
      <div className="sec">{title}</div>
      <p className="note">{note}</p>
      <ul className="findings">
        {sorted.map((f) => {
          const key = findingKey(f)
          return (
            <li key={key} className={`finding${selected === key ? ' sel' : ''}`}>
              <button className="fhead" onClick={() => onSelect(f)} aria-pressed={selected === key} title={f.points.length ? 'Show in viewport' : undefined}>
                <span className={CLASS[f.status]} aria-label={f.status}>{ICON[f.status]}</span>
                <span className="ft">{f.title}</span>
                <span className="fv">{valueText(f, units)}</span>
              </button>
              <div className="fd">
                {f.detail}
                {f.refs.length > 0 && (
                  <span className="frefs">
                    {f.refs.map((r) => <button key={r} className="linkish" onClick={() => onRef(r)}>{r}</button>)}
                  </span>
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </>
  )
}
