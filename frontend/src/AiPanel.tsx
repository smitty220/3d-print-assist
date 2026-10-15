import { useState } from 'react'
import type { AiStatus, AiStep, ValidationReport } from './api'
import { toDisplay, UNIT_LABEL, type Units } from './units'

const KIND_ICON: Record<AiStep['kind'], string> = { note: '›', build: '⚙', template: '▤', finish: '✔', error: '✖' }

interface Props {
  status: AiStatus | null
  report: ValidationReport | null
  hasDesign: boolean
  units: Units
  onGenerate: (instruction: string, rounds: number) => void
  onCancel: () => void
  onRevert: () => void
}

export function AiPanel({ status, report, hasDesign, units, onGenerate, onCancel, onRevert }: Props) {
  const [instruction, setInstruction] = useState('')
  const [rounds, setRounds] = useState(6)
  const job = status?.job ?? null
  const running = job?.status === 'running'
  const blocked = report != null && !report.preflight_ready
  const mass = (kg: number) => `${toDisplay(kg, 'mass', units).toFixed(3)} ${UNIT_LABEL[units].mass}`

  return (
    <>
      <div className="ttl">
        <b>Generate with AI</b>
        <span>{status ? status.model : 'checking…'}</span>
      </div>
      {status && !status.key && (
        <p className="errbox">No API key yet. Put <code>ANTHROPIC_API_KEY=…</code> in <code>backend/.env</code> (or export it in your shell) and restart the backend.</p>
      )}
      {blocked && <p className="errbox">Pre-flight has failures. Fix the intent first; the AI only starts on an intent a part can satisfy.</p>}

      <div className="field">
        <label htmlFor="ai-ask">Request (optional)</label>
        <textarea id="ai-ask" rows={3} value={instruction} disabled={running} onChange={(e) => setInstruction(e.target.value)}
          placeholder={hasDesign ? 'e.g. get under the mass target without thinning the ears' : 'e.g. design the bracket; keep it simple to machine'} />
      </div>
      <div className="sp-row">
        <label className="ctl" htmlFor="ai-rounds">
          Builds
          <select id="ai-rounds" value={rounds} disabled={running} onChange={(e) => setRounds(+e.target.value)}>
            {[3, 6, 9, 12].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        {running ? (
          <button className="btn danger" onClick={onCancel}>Stop</button>
        ) : (
          <button className="btn primary" disabled={!status?.key || blocked} onClick={() => onGenerate(instruction, rounds)}>
            {hasDesign ? 'Improve design' : 'Generate design'}
          </button>
        )}
      </div>
      <p className="note">
        {hasDesign
          ? 'Claude starts from the current feature tree and keeps every locked feature as you set it.'
          : 'Claude picks a starting template, then revises it against the validator.'}{' '}
        Each candidate shows in the viewport as it is built; the best one is kept.
      </p>

      {job && (
        <>
          <div className="sec">
            {running ? `Working · build ${job.rounds} of ${job.max_rounds}` : `Last run · ${job.status}`}
            {running && <span className="spin inline" aria-label="working" />}
          </div>
          {job.instruction && <p className="note">“{job.instruction}”</p>}
          <ol className="ailog">
            {job.steps.map((s, i) => (
              <li key={i} className={`ai-${s.kind}`}>
                <span className="ico" aria-hidden="true">{KIND_ICON[s.kind]}</span>
                <span className="tx">
                  {s.kind === 'build' && <b>Build {s.round}: </b>}
                  {s.text}
                  {s.counts && (
                    <span className="aimeta">
                      <span className={s.counts.fail ? 'no' : 'ok'}>✖ {s.counts.fail}</span> <span className="wn">⚠ {s.counts.warn}</span>
                      {s.mass != null && ` · ${mass(s.mass)}`}
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ol>
          {!running && (
            <>
              {job.best && (
                <p className="note">
                  Kept build {job.best.round}: {job.best.counts.fail} failing, {job.best.counts.warn} warnings, {mass(job.best.mass)}.
                </p>
              )}
              {job.error && <p className="errbox">{job.error}</p>}
              <div className="sp-row">
                <button className="btn" disabled={!job.can_revert} onClick={onRevert} title="Put back the design you had before this run">Undo AI run</button>
                <span className="nm">{(job.tokens.input / 1000).toFixed(0)}k in · {(job.tokens.output / 1000).toFixed(1)}k out</span>
              </div>
            </>
          )}
        </>
      )}
    </>
  )
}
