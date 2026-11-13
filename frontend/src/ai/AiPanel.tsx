// The AI panel (always on the right). Two modes: Claude Code drives the app over MCP (no key needed; this panel shows
// what it does, live), or the in-app chat calls the Claude API (for the home server).
import { useEffect, useRef, useState } from 'react'
import { api, type Activity, type AiStatus, type Concept, type ImageInfo } from '../api'
import { Icon } from '../ui/icons'
import type { usePanelLayout } from '../ui/usePanelLayout'

type PanelLayout = ReturnType<typeof usePanelLayout>

interface Props {
  status: AiStatus | null
  activity: Activity[]
  images: ImageInfo[]
  context: string | null
  concept: Concept | null // the newest
  layout: PanelLayout
  zIndex?: number
  onFocusPanel?: () => void
  onOpenConcept: (id: string) => void
  onUploadImage: (file: File) => Promise<ImageInfo | undefined>
  onDeleteImage: (id: string) => Promise<void>
  onSettings?: (patch: { ai_mode?: 'mcp' | 'api'; ai_provider?: 'auto' | 'agy' | 'gemini' | 'anthropic'; ai_model?: string }) => void
  onSend: (text: string) => void
  onCancel: () => void
  onReset: () => void
  onRevert: () => void
}

const time = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

function defaultMcpUrl(): string {
  try {
    const saved = localStorage.getItem('pa.mcpUrl')
    if (saved) return saved
  } catch { /* ignore */ }
  if (typeof window === 'undefined') return 'http://127.0.0.1:8010'
  return window.location.port === '5183'
    ? `http://${window.location.hostname}:8010`
    : window.location.origin
}

export function AiPanel({ status, activity, images, context, concept, layout, zIndex, onFocusPanel, onOpenConcept, onUploadImage, onDeleteImage, onSettings, onSend, onCancel, onReset, onRevert }: Props) {
  const mode = status?.mode ?? 'mcp'
  const [text, setText] = useState('')
  const [attached, setAttached] = useState<string[]>([])
  const [uploading, setUploading] = useState(false)
  const [mcpUrl, setMcpUrl] = useState(defaultMcpUrl)
  const [copied, setCopied] = useState(false)
  const end = useRef<HTMLDivElement>(null)
  const imgInput = useRef<HTMLInputElement>(null)
  const chat = status?.chat
  useEffect(() => { end.current?.scrollIntoView({ block: 'end' }) }, [chat?.log.length, activity.length])
  // The AI's changes, and the user's answers to its concepts (the conversation about what to build).
  const ai = activity.filter((a) => a.actor !== 'user' || a.text.startsWith('Concept '))
  const uploadFiles = async (files: FileList | File[]) => {
    const list = Array.from(files).filter((f) => f.type.startsWith('image/') || /\.(png|jpe?g|webp)$/i.test(f.name))
    if (!list.length) return
    setUploading(true)
    try {
      for (const f of list) {
        const img = await onUploadImage(f)
        if (img) setAttached((prev) => (prev.includes(img.id) ? prev : [...prev, img.id]))
      }
    } finally {
      setUploading(false)
    }
  }
  const toggleAttach = (id: string) => setAttached((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
  const send = () => {
    const t = text.trim()
    if (!t) return
    const parts = [t]
    if (attached.length) parts.push(`(Reference image(s) attached: ${attached.join(', ')} — inspect with look_at_image.)`)
    if (context) parts.push(`(The user has ${context} selected.)`)
    onSend(parts.join('\n\n'))
    setText('')
    setAttached([])
  }
  const refStrip = (
    <div className="row" style={{ padding: '6px 12px', gap: 6, borderTop: '1px solid var(--line)', background: 'var(--panel-2)', flexWrap: 'wrap', alignItems: 'center' }}>
      <input
        ref={imgInput}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        multiple
        hidden
        onChange={(e) => { if (e.target.files) uploadFiles(e.target.files); e.target.value = '' }}
      />
      <button type="button" className="btn sm" disabled={uploading} onClick={() => imgInput.current?.click()}
        title="Upload a reference photo (PNG, JPG, WebP) so the AI can inspect it">
        {uploading ? 'Uploading…' : '+ Photo'}
      </button>
      {images.map((img) => {
        const on = mode !== 'mcp' && attached.includes(img.id)
        return (
          <span
            key={img.id}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 2,
              border: `1px solid ${on ? 'var(--accent)' : 'var(--line)'}`,
              borderRadius: 6,
              background: on ? 'color-mix(in srgb, var(--accent) 18%, transparent)' : 'var(--panel)',
            }}
          >
            <button
              type="button"
              className="btn sm"
              onClick={() => mode !== 'mcp' && toggleAttach(img.id)}
              title={mode === 'mcp' ? `${img.id}: ${img.prompt} (ask MCP client to look_at_image ${img.id})` : `${on ? 'Detach' : 'Attach'} ${img.id}: ${img.prompt}`}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 5px', border: 0, background: 'transparent' }}
            >
              <img src={api.imageUrl(img.id)} alt={img.id} style={{ width: 18, height: 18, objectFit: 'cover', borderRadius: 3 }} />
              <span>{img.id}</span>
            </button>
            <button
              type="button"
              aria-label={`Remove ${img.id}`}
              title={`Remove ${img.id}`}
              onClick={async (e) => {
                e.stopPropagation()
                setAttached((prev) => prev.filter((x) => x !== img.id))
                await onDeleteImage(img.id)
              }}
              style={{ border: 0, background: 'transparent', color: 'var(--muted)', cursor: 'pointer', padding: '0 5px 0 2px', fontSize: 13, lineHeight: 1 }}
            >
              ×
            </button>
          </span>
        )
      })}
    </div>
  )

  const isCollapsed = layout.collapsed && !layout.floating
  return (
    <aside
      className={`side ai${isCollapsed ? ' collapsed' : ''}${layout.floating ? ' floating' : ''}`}
      style={layout.floating ? { left: layout.box.x, top: layout.box.y, width: layout.box.w, height: layout.box.h, zIndex } : undefined}
      onPointerDown={onFocusPanel}
      aria-label="AI assistant"
    >
      {isCollapsed ? (
        <button type="button" className="side-rail" onClick={() => layout.setCollapsed(false)} aria-expanded={false}
          aria-label="Expand AI assistant panel" title="Expand AI assistant panel">
          <Icon name="chevL" size={14} />
          <span className="side-rail-label">AI assistant</span>
        </button>
      ) : null}
      <div className="side-body" hidden={isCollapsed}>
        <div className="aihead">
          {layout.floating ? (
            <button type="button" className="side-toggle" onClick={() => layout.setFloating(false)}
              aria-label="Dock AI assistant panel" title="Dock AI assistant panel">
              <Icon name="dock" size={14} />
            </button>
          ) : (
            <button type="button" className="side-toggle" onClick={() => layout.setCollapsed(true)} aria-expanded={true}
              aria-label="Collapse AI assistant panel" title="Collapse AI assistant panel">
              <Icon name="chevR" size={14} />
            </button>
          )}
          <button type="button" className={`side-grip${layout.dragging ? ' dragging' : ''}`}
            aria-label="Move AI assistant panel. Drag to float or use arrow keys when floating."
            title="Drag to float / move panel" {...layout.moveHandleProps}>
            <Icon name="grip" size={14} />
          </button>
          <h2>AI assistant</h2>
        </div>
      <div className="row sm" style={{ padding: '8px 12px', gap: 10, borderBottom: '1px solid var(--line)', background: 'var(--panel-2)', boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', flexDirection: 'column', flex: '1 1 0', minWidth: 0, gap: 3 }}>
          <label htmlFor="ai-provider-select" style={{ fontSize: 10, textTransform: 'uppercase', opacity: 0.65, letterSpacing: '0.04em' }}>Assistant</label>
          <select
            id="ai-provider-select"
            value={mode === 'mcp' ? 'mcp' : (status?.settings_provider || 'auto')}
            onChange={(e) => {
              const v = e.target.value
              if (v === 'mcp') {
                onSettings?.({ ai_mode: 'mcp' })
              } else {
                onSettings?.({ ai_mode: 'api', ai_provider: v as 'auto' | 'agy' | 'gemini' | 'anthropic', ai_model: '' })
              }
            }}
            style={{ fontSize: 12, padding: '4px 8px', width: '100%', minWidth: 0, boxSizing: 'border-box' }}
          >
            <option value="auto">Auto ({status?.provider === 'agy' ? 'Antigravity' : status?.provider === 'gemini' ? 'Gemini' : 'Claude'})</option>
            <option value="agy">Antigravity CLI (Subscription)</option>
            <option value="gemini">Google Gemini (API Key)</option>
            <option value="anthropic">Anthropic Claude (API Key)</option>
            <option value="mcp">External MCP (Terminal / IDE)</option>
          </select>
        </div>
        {mode !== 'mcp' && (
          <div style={{ display: 'flex', flexDirection: 'column', flex: '1.25 1 0', minWidth: 0, gap: 3 }}>
            <label htmlFor="ai-model-select" style={{ fontSize: 10, textTransform: 'uppercase', opacity: 0.65, letterSpacing: '0.04em' }}>Model</label>
            {(() => {
              const models = status?.available_models?.[status?.provider || 'gemini'] || []
              const picked = status?.settings_model && models.some((m) => m.id === status.settings_model)
                ? status.settings_model
                : (status?.model && models.some((m) => m.id === status.model) ? status.model : (models[0]?.id ?? ''))
              return (
                <select
                  id="ai-model-select"
                  value={picked}
                  onChange={(e) => {
                    onSettings?.({ ai_model: e.target.value })
                  }}
                  style={{ fontSize: 12, padding: '4px 8px', width: '100%', minWidth: 0, boxSizing: 'border-box' }}
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>{m.label}</option>
                  ))}
                </select>
              )
            })()}
          </div>
        )}
      </div>
      {context && <div className="ctx">Context: {context}</div>}
      {concept && (
        <div className="ccard">
          <div className="row">
            <b>{concept.id} · {concept.title}</b><span className="spacer" />
            <span className={`chip cstat ${concept.status}`}>{concept.status === 'pending' ? 'Waiting for you' : concept.status === 'approved' ? 'Approved' : 'Changes asked'}</span>
          </div>
          <img src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(concept.svg)}`} alt={`Concept drawing: ${concept.title}`}
            onClick={() => onOpenConcept(concept.id)} />
          <button className="btn sm" onClick={() => onOpenConcept(concept.id)}>{concept.status === 'pending' ? 'Answer' : 'Open'}</button>
        </div>
      )}

      {mode === 'mcp' ? (() => {
        const cleanUrl = (mcpUrl.trim() || defaultMcpUrl()).replace(/\/+$/, '')
        const remoteMcpUrl = `${cleanUrl}/mcp/`
        const mcpJson = JSON.stringify({
          mcpServers: {
            '3d-print-assist': {
              type: 'http',
              url: remoteMcpUrl,
            },
          },
        }, null, 2)
        return (
          <>
            <div className="log">
              {ai.length === 0 && (
                <div className="intro">
                  <p>Connect Claude Code, Gemini CLI, or any MCP client to this app. You watch each change land here and in the tree, and can edit alongside it.</p>
                  <p>To chat directly with <b>{status?.provider === 'gemini' ? 'Google Gemini' : status?.provider === 'agy' ? 'Antigravity' : 'Claude'}</b> in this window, pick an Assistant from the dropdown above.</p>
                </div>
              )}
              <details className="intro" open={ai.length === 0} style={{ marginTop: 6 }}>
                <summary style={{ cursor: 'pointer', fontWeight: 600 }}>MCP connection (no local repo needed)</summary>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
                  <label htmlFor="mcp-url-input" className="nm">Server URL (auto-detected; edit if connecting over LAN/VPN):</label>
                  <div className="row" style={{ gap: 6 }}>
                    <input
                      id="mcp-url-input"
                      type="url"
                      value={mcpUrl}
                      onChange={(e) => {
                        setMcpUrl(e.target.value)
                        try { localStorage.setItem('pa.mcpUrl', e.target.value) } catch { /* ignore */ }
                      }}
                      placeholder="http://127.0.0.1:8010"
                      style={{ flex: 1, minWidth: 0, fontSize: 12, padding: '4px 8px' }}
                    />
                    <button
                      type="button"
                      className="btn sm"
                      onClick={() => {
                        try { localStorage.removeItem('pa.mcpUrl') } catch { /* ignore */ }
                        setMcpUrl(defaultMcpUrl())
                      }}
                      title="Reset to detected URL"
                    >
                      Auto
                    </button>
                  </div>
                  <pre style={{ margin: 0, padding: 8, borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--line)', fontSize: 11, overflowX: 'auto' }}>{mcpJson}</pre>
                  <div className="row" style={{ gap: 6 }}>
                    <button
                      type="button"
                      className="btn sm"
                      onClick={() => {
                        navigator.clipboard?.writeText(mcpJson).then(() => {
                          setCopied(true)
                          setTimeout(() => setCopied(false), 2000)
                        }).catch(() => {})
                      }}
                    >
                      {copied ? 'Copied!' : 'Copy .mcp.json'}
                    </button>
                    <span className="nm">Endpoint: <code>{remoteMcpUrl}</code></span>
                  </div>
                </div>
              </details>
              {ai.map((a, i) => (
                <div key={i} className={`entry ${a.ok ? 'tool' : 'error'}`}>
                  <span className="who">{a.actor}</span><span className="when">{time(a.at)}</span>
                  <div>{a.text}</div>
                </div>
              ))}
              <div ref={end} />
            </div>
            {refStrip}
            <div className="aifoot">
              <button className="btn" onClick={onRevert} disabled={!ai.some((a) => a.ok && a.actor !== 'user')}>Revert last AI change</button>
            </div>
          </>
        )
      })() : (
        <>
          <div className="log">
            {!status?.key && (
              <div className="intro">
                {status?.provider === 'agy' ? (
                  <p>Antigravity CLI (<code>agy</code>) was not found. Make sure <code>agy</code> is installed (e.g. <code>~/.local/bin/agy</code>).</p>
                ) : (
                  <p>
                    No {status?.provider === 'gemini' ? 'Gemini' : 'Claude'} API key yet. Put{' '}
                    <code>{status?.provider === 'gemini' ? 'GEMINI_API_KEY=…' : 'ANTHROPIC_API_KEY=…'}</code> (or{' '}
                    <code>{status?.provider === 'gemini' ? 'ANTHROPIC_API_KEY=…' : 'GEMINI_API_KEY=…'}</code>) in{' '}
                    <code>backend/.env</code>; it is read on the next message, no restart needed. Model: {status?.model ?? '…'}.
                  </p>
                )}
                <p>Or choose <b>External MCP (Terminal / IDE)</b> above to drive the app from Claude Code.</p>
              </div>
            )}
            {chat?.log.map((e, i) => (
              <div key={i} className={`entry ${e.role}${e.ok ? '' : ' error'}`}>
                {e.role === 'tool' ? <><span className="who">{e.tool}</span><div className="toolout">{e.text}</div></> : <div>{e.text}</div>}
              </div>
            ))}
            {chat?.running && <div className="entry assistant working">Working…</div>}
            <div ref={end} />
          </div>
          {refStrip}
          <div className="aifoot col">
            <label htmlFor="ai-msg" className="nm">Ask for a change</label>
            <div className="row">
              <textarea id="ai-msg" rows={3} value={text} onChange={(e) => setText(e.target.value)} disabled={chat?.running}
                placeholder="e.g. make the walls 2 mm and twist it 90° (paste or + Photo to attach reference images)"
                onPaste={(e) => {
                  const files = e.clipboardData?.files
                  if (files?.length && Array.from(files).some((f) => f.type.startsWith('image/'))) {
                    e.preventDefault()
                    uploadFiles(files)
                  }
                }}
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send() }} />
              {chat?.running
                ? <button className="btn" onClick={onCancel}>Stop</button>
                : <button className="send" aria-label="Send" onClick={send} disabled={!text.trim()}><Icon name="send" size={18} width={2} /></button>}
            </div>
            <div className="row nm">
              {chat ? `${Math.round(((chat.tokens.input ?? 0) + (chat.tokens.output ?? 0)) / 1000)}k tokens` : ''}
              {status?.model ? <span className="dim" style={{ marginLeft: 6, fontSize: '0.85em', opacity: 0.7 }}>({status.model})</span> : null}
              <span className="spacer" />
              <button className="link" onClick={onRevert}>Revert last AI change</button>
              <button className="link" onClick={onReset} disabled={chat?.running}>New conversation</button>
            </div>
          </div>
        </>
      )}
      </div>
      {!isCollapsed && !layout.floating && (
        <div className="side-resizer right" role="separator" aria-orientation="vertical"
          aria-label="Resize AI assistant panel. Use left and right arrow keys or double-click to reset."
          title="Drag to resize (double-click to reset)" tabIndex={0} {...layout.dockResizeProps} />
      )}
      {layout.floating && (
        <div className="float-resizer" role="separator" aria-label="Resize floating AI assistant panel"
          title="Drag to resize" {...layout.floatResizeProps} />
      )}
    </aside>
  )
}
