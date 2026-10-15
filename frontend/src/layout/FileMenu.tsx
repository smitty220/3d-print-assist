// File menu (and the Edit actions beside it): traditional document handling over the design's folder.
import { useEffect, useRef, useState } from 'react'
import type { FileState } from '../api'
import { Icon } from '../ui/icons'

export type FileAction = 'new' | 'open' | 'save' | 'save-version' | 'save-as' | 'rename' | 'revert' | 'history' | 'close' | 'delete' | 'export' | 'reveal'

const MAC = /Mac|iPhone|iPad/.test(navigator.platform)
const K = (k: string) => (MAC ? `⌘${k}` : `Ctrl+${k}`)
const ITEMS: ({ id: FileAction; label: string; key?: string; needsDirty?: boolean } | null)[] = [
  { id: 'new', label: 'New…' }, { id: 'open', label: 'Open…', key: K('O') }, null,
  { id: 'save', label: 'Save', key: K('S') }, { id: 'save-version', label: 'Save version…' }, { id: 'save-as', label: 'Save as…', key: `⇧${K('S')}` },
  { id: 'revert', label: 'Revert to saved', needsDirty: true }, null,
  { id: 'rename', label: 'Rename…' }, { id: 'history', label: 'Saved versions…' }, { id: 'reveal', label: 'Show in Finder' }, null,
  { id: 'export', label: 'Export…' }, null,
  { id: 'close', label: 'Close' }, { id: 'delete', label: 'Move to Trash…' },
]

export function FileMenu({ file, onAction, onUndo, onRedo }: { file: FileState; onAction: (a: FileAction) => void; onUndo: () => void; onRedo: () => void }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    window.addEventListener('mousedown', close)
    return () => window.removeEventListener('mousedown', close)
  }, [open])
  return (
    <div className="filebar">
      <div className="fmenu" ref={ref}>
        <button className="fbtn" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}>File</button>
        {open && (
          <div className="fdrop" role="menu">
            {ITEMS.map((it, i) => it === null ? <hr key={i} /> : (
              <button key={it.id} role="menuitem" disabled={it.needsDirty && !file.dirty} onClick={() => { setOpen(false); onAction(it.id) }}>
                <span>{it.label}</span>{it.key && <span className="kb">{it.key}</span>}
              </button>
            ))}
          </div>
        )}
      </div>
      <button className="ibtn" onClick={onUndo} disabled={!file.undo} title={file.undo ? `Undo ${file.undo} (${K('Z')})` : 'Nothing to undo'} aria-label="Undo"><Icon name="undo" size={16} /></button>
      <button className="ibtn flip" onClick={onRedo} disabled={!file.redo} title={file.redo ? `Redo ${file.redo} (⇧${K('Z')})` : 'Nothing to redo'} aria-label="Redo"><Icon name="undo" size={16} /></button>
    </div>
  )
}
