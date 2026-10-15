// The File menu's dialogs: a name, unsaved changes, opening a design, and saved versions. Never browser alerts.
import { useEffect, useState } from 'react'
import { api, type DesignInfo, type Version } from '../api'

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

export function NameModal({ title, label, initial, ok, onOk, onClose }: { title: string; label: string; initial: string; ok: string; onOk: (v: string) => void; onClose: () => void }) {
  const [v, setV] = useState(initial)
  return (
    <Modal title={title} onClose={onClose}>
      <form className="dbody" onSubmit={(e) => { e.preventDefault(); if (v.trim()) onOk(v.trim()) }}>
        <div className="xfield"><label htmlFor="nm-in">{label}</label>
          <input id="nm-in" autoFocus value={v} onChange={(e) => setV(e.target.value)} onFocus={(e) => e.target.select()} />
        </div>
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

export function OpenModal({ current, onOpen, onDelete, onClose }: { current: string | null; onOpen: (slug: string) => void; onDelete: (d: DesignInfo) => void; onClose: () => void }) {
  const [list, setList] = useState<DesignInfo[] | null>(null)
  useEffect(() => { api.projects().then(setList).catch(() => setList([])) }, [])
  return (
    <Modal title="Open design" onClose={onClose} wide>
      <div className="dbody"><DesignGrid list={list} current={current} onOpen={onOpen} onDelete={onDelete} /></div>
    </Modal>
  )
}

export function DesignGrid({ list, current, onOpen, onDelete }: { list: DesignInfo[] | null; current: string | null; onOpen: (slug: string) => void; onDelete?: (d: DesignInfo) => void }) {
  if (!list) return <p className="note">Loading…</p>
  if (!list.length) return <p className="note">No designs yet.</p>
  return (
    <ul className="grid">
      {list.map((d) => (
        <li key={d.slug} className={d.slug === current ? 'cur' : ''}>
          <button className="card" onClick={() => onOpen(d.slug)} disabled={d.slug === current}>
            {d.thumbnail ? <img src={api.thumbnailUrl(d.slug, d.modified)} alt="" /> : <span className="noimg">no picture until saved</span>}
            <b>{d.name}</b>
            <span className="nm">{new Date(d.modified * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}{d.recovery ? ' · unsaved changes kept' : ''}</span>
          </button>
          {onDelete && d.slug !== current && <button className="link danger" onClick={() => onDelete(d)}>Move to Trash</button>}
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

export function HistoryModal({ onRestore, onClose }: { onRestore: (v: Version) => void; onClose: () => void }) {
  const [list, setList] = useState<Version[] | null>(null)
  useEffect(() => { api.history().then(setList).catch(() => setList([])) }, [])
  return (
    <Modal title="Saved versions" onClose={onClose}>
      <div className="dbody">
        <p className="note">Restoring a version loads it as unsaved changes: look it over, then save (or undo to go back).</p>
        <ul className="versions">
          {(list ?? []).map((v) => (
            <li key={v.id}>
              <div><b>{v.message}</b><div className="nm">{new Date(v.at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })} · {v.id}</div></div>
              <button className="btn sm" onClick={() => onRestore(v)}>Restore</button>
            </li>
          ))}
        </ul>
      </div>
    </Modal>
  )
}
