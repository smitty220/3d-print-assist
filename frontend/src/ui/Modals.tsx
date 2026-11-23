// The File menu's dialogs: a name, unsaved changes, opening a design, and saved versions. Never browser alerts.
import { useEffect, useState } from 'react'
import { api, type CreationMode, type DesignInfo, type Version } from '../api'
import { Spinner } from './Spinner'

function Modal({ title, onClose, children, wide }: { title: string; onClose: () => void; children: React.ReactNode; wide?: boolean }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])
  return (
    <div className="scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <section className={`modal${wide ? ' wide' : ''}`} role="dialog" aria-modal="true" aria-label={title}>
        <div className="dhead"><h2>{title}</h2><span className="spacer" /><button className="x" onClick={onClose} aria-label="Close">×</button></div>
        {children}
      </section>
    </div>
  )
}

export function NameModal({ title, label, initial, ok, showModePicker, onOk, onClose }: {
  title: string; label: string; initial: string; ok: string; showModePicker?: boolean
  onOk: (v: string, mode?: CreationMode) => void; onClose: () => void
}) {
  const [v, setV] = useState(initial)
  const [mode, setMode] = useState<CreationMode>('engineer')
  return (
    <Modal title={title} onClose={onClose}>
      <form className="dbody" onSubmit={(e) => { e.preventDefault(); if (v.trim()) onOk(v.trim(), showModePicker ? mode : undefined) }}>
        <div className="xfield"><label htmlFor="nm-in">{label}</label>
          <input id="nm-in" autoFocus value={v} onChange={(e) => setV(e.target.value)} onFocus={(e) => e.target.select()} />
        </div>
        {showModePicker && (
          <fieldset style={{ border: '1px solid var(--line)', borderRadius: 8, padding: '8px 10px', margin: '4px 0 0' }}>
            <legend style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em', opacity: 0.75, padding: '0 4px' }}>Creation Mode</legend>
            <div className="row" style={{ gap: 8 }}>
              <button type="button" className={`btn${mode === 'engineer' ? ' primary' : ''}`} style={{ flex: 1, textAlign: 'left', padding: '8px 10px' }}
                aria-pressed={mode === 'engineer'} onClick={() => setMode('engineer')}>
                <div><b>Engineer Mode</b></div>
                <div className="nm" style={{ opacity: 0.85, fontSize: 11 }}>Math, extrusions &amp; exact CAD parts</div>
              </button>
              <button type="button" className={`btn${mode === 'character' ? ' primary' : ''}`} style={{ flex: 1, textAlign: 'left', padding: '8px 10px' }}
                aria-pressed={mode === 'character'} onClick={() => setMode('character')}>
                <div><b>Character Mode</b></div>
                <div className="nm" style={{ opacity: 0.85, fontSize: 11 }}>Molding clay: figurines &amp; organic scenes</div>
              </button>
            </div>
          </fieldset>
        )}
        <div className="dfoot flush"><span className="spacer" /><button type="button" className="btn" onClick={onClose}>Cancel</button>
          <button type="submit" className="btn primary" disabled={!v.trim()}>{ok}</button></div>
      </form>
    </Modal>
  )
}

export function UnsavedModal({ name, onSave, onDiscard, onClose }: { name: string; onSave: () => void; onDiscard: () => void; onClose: () => void }) {
  return (
    <Modal title="Unsaved changes" onClose={onClose}>
      <div className="dbody"><p>Save the changes to <b>{name}</b> first?</p><p className="note">If you don't save, they are lost.</p></div>
      <div className="dfoot">
        <button className="btn danger" onClick={onDiscard}>Don't save</button><span className="spacer" />
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" autoFocus onClick={onSave}>Save</button>
      </div>
    </Modal>
  )
}

export function OpenModal({ current, busySlug, onOpen, onDelete, onClose }: { current: string | null; busySlug?: string | null; onOpen: (slug: string) => void; onDelete: (d: DesignInfo) => void; onClose: () => void }) {
  const [list, setList] = useState<DesignInfo[] | null>(null)
  useEffect(() => { api.projects().then(setList).catch(() => setList([])) }, [])
  return (
    <Modal title="Open design" onClose={onClose} wide>
      <div className="dbody">
        {busySlug && <p className="note spin-row" role="status"><Spinner size={14} decorative /> Opening design…</p>}
        <DesignGrid list={list} current={current} busySlug={busySlug} onOpen={onOpen} onDelete={onDelete} />
      </div>
    </Modal>
  )
}

export function DesignGrid({ list, current, busySlug, onOpen, onDelete }: { list: DesignInfo[] | null; current: string | null; busySlug?: string | null; onOpen: (slug: string) => void; onDelete?: (d: DesignInfo) => void }) {
  if (!list) return <p className="note spin-row" role="status"><Spinner size={15} label="Loading designs" /> Loading designs…</p>
  if (!list.length) return <p className="note">No designs yet.</p>
  return (
    <ul className="grid">
      {list.map((d) => (
        <li key={d.slug} className={d.slug === current ? 'cur' : ''}>
          <button type="button" className="card" onClick={() => onOpen(d.slug)} disabled={d.slug === current || Boolean(busySlug)}>
            {d.thumbnail ? <img src={api.thumbnailUrl(d.slug, d.modified)} alt="" /> : <span className="noimg">no picture until saved</span>}
            <b className={d.slug === busySlug ? 'spin-row' : undefined}>
              {d.slug === busySlug && <Spinner size={13} decorative />}
              {d.slug === busySlug ? `${d.name} (Opening…)` : d.name}
            </b>
            <span className="nm">{new Date(d.modified * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}{d.recovery ? ' · unsaved changes kept' : ''}</span>
          </button>
          {onDelete && d.slug !== current && <button type="button" className="link danger" disabled={Boolean(busySlug)} onClick={() => onDelete(d)}>Move to Trash</button>}
        </li>
      ))}
    </ul>
  )
}

export function ConfirmModal({ title, text, ok, onOk, onClose }: { title: string; text: string; ok: string; onOk: () => void; onClose: () => void }) {
  return (
    <Modal title={title} onClose={onClose}>
      <div className="dbody"><p>{text}</p></div>
      <div className="dfoot"><span className="spacer" /><button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn danger" onClick={onOk}>{ok}</button></div>
    </Modal>
  )
}

export function HistoryModal({ onRestore, onClose }: { onRestore: (v: Version) => Promise<void> | void; onClose: () => void }) {
  const [list, setList] = useState<Version[] | null>(null)
  const [restoringId, setRestoringId] = useState<string | null>(null)
  useEffect(() => { api.history().then(setList).catch(() => setList([])) }, [])
  const handleRestore = async (v: Version) => {
    if (restoringId) return
    setRestoringId(v.id)
    try {
      await onRestore(v)
    } finally {
      setRestoringId(null)
    }
  }
  return (
    <Modal title="Saved versions" onClose={() => { if (!restoringId) onClose() }}>
      <div className="dbody">
        <p className="note">Restoring a version loads it as unsaved changes: look it over, then save (or undo to go back).</p>
        {!list ? (
          <p className="note spin-row" role="status"><Spinner size={15} label="Loading saved versions" /> Loading saved versions…</p>
        ) : !list.length ? (
          <p className="note">No saved versions yet.</p>
        ) : (
          <ul className="versions">
            {list.map((v) => {
              const busy = restoringId === v.id
              return (
                <li key={v.id}>
                  <div><b>{v.message}</b><div className="nm">{new Date(v.at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })} · {v.id}</div></div>
                  <button className="btn sm" disabled={Boolean(restoringId)} onClick={() => handleRestore(v)}>
                    {busy && <Spinner size={12} decorative />}
                    {busy ? 'Restoring…' : 'Restore'}
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </Modal>
  )
}

const FORMATS = [
  { id: '3mf', name: 'Bambu Studio project (.3mf)', note: 'For printing: each body a part in its filament slot, the colours loaded.' },
  { id: 'stl', name: 'STL mesh (.stl)', note: 'One colour, for other slicers or sharing.' },
  { id: 'step', name: 'STEP solid (.step)', note: 'Exact geometry, for other CAD programs.' },
] as const
export type ExportFormat = (typeof FORMATS)[number]['id']

/** File > Export: pick a format; the file downloads and is also kept in the design's exports folder. */
export function ExportModal({ why, onUpdate, onExport, onMore, onClose }: {
  why: string | null; onUpdate: () => void; onExport: (f: ExportFormat) => Promise<void>; onMore: () => void; onClose: () => void
}) {
  const [fmt, setFmt] = useState<ExportFormat>('3mf')
  const [busy, setBusy] = useState(false)
  return (
    <Modal title="Export" onClose={onClose}>
      <div className="dbody">
        <div className="fmts" role="radiogroup" aria-label="Format">
          {FORMATS.map((f) => (
            <label key={f.id} className={`fmt${fmt === f.id ? ' on' : ''}`}>
              <input type="radio" name="fmt" value={f.id} checked={fmt === f.id} onChange={() => setFmt(f.id)} />
              <span><b>{f.name}</b><span className="nm">{f.note}</span></span>
            </label>
          ))}
        </div>
        {why && <p className="note wn">{why}</p>}
        <p className="note">The file downloads and a copy is kept in the design's exports folder. Colours, variants and print checks
          are on the <button className="link" onClick={onMore}>Colour &amp; Export tab</button>.</p>
      </div>
      <div className="dfoot"><span className="spacer" /><button className="btn" onClick={onClose}>Cancel</button>
        {why === 'Update the model first'
          ? <button className="btn primary" onClick={onUpdate}>Update</button>
          : <button className="btn primary" autoFocus disabled={!!why || busy}
              onClick={() => { setBusy(true); onExport(fmt).finally(() => setBusy(false)) }}>
              {busy && <Spinner size={13} decorative />}
              {busy ? 'Exporting…' : 'Export'}
            </button>}
      </div>
    </Modal>
  )
}
