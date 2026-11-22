// The AI panel (always on the right). Two modes: Claude Code drives the app over MCP (no key needed; this panel shows
// what it does, live), or the in-app chat calls the Claude API (for the home server).
import { useEffect, useRef, useState } from 'react'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { api, type Activity, type AiStatus, type Concept, type CreationMode, type ImageInfo, type MlStatus, type SculptMesh, type SculptPrep } from '../api'
import { Icon } from '../ui/icons'
import type { usePanelLayout } from '../ui/usePanelLayout'

type PanelLayout = ReturnType<typeof usePanelLayout>

interface Props {
  status: AiStatus | null
  activity: Activity[]
  images: ImageInfo[]
  sculpts?: SculptMesh[]
  mlStatus?: MlStatus | null
  creationMode?: CreationMode
  directBuild?: boolean
  sculpting?: boolean
  context: string | null
  concept: Concept | null // the newest
  conceptOpen?: boolean
  onAnswerConcept?: (id: string, decision: 'approve' | 'revise', note: string) => void
  onBuildConcept?: (concept: Concept, note: string) => void
  layout: PanelLayout
  zIndex?: number
  onFocusPanel?: () => void
  onOpenConcept: (id: string) => void
  onOpenPhoto?: (id: string) => void
  onUploadImage: (file: File) => Promise<ImageInfo | undefined>
  onDeleteImage: (id: string) => Promise<void>
  onSetupMl?: () => void
  onCancelMl?: () => void
  onUpdateSculptPrep?: (id: string, patch: Partial<SculptPrep>) => void
  onSettings?: (patch: { ai_mode?: 'mcp' | 'api'; ai_provider?: 'auto' | 'agy' | 'gemini' | 'anthropic'; ai_model?: string; creation_mode?: CreationMode; direct_build?: boolean; character_multicolor?: boolean }) => void
  onSend: (text: string) => void
  onCancel: () => void
  onReset: () => void
  onRevert: () => void
}

const time = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

function renderMarkdown(text: string): string {
  const html = marked.parse(text, { async: false, breaks: true, gfm: true })
  return DOMPurify.sanitize(html)
}

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

const TOOL_LABELS: Record<string, string> = {
  get_guide: 'reading guide',
  get_model: 'inspecting model',
  get_feature_catalog: 'checking features',
  apply_changes: 'updating model',
  update: 'rebuilding geometry',
  render_preview: 'rendering preview',
  run_checks: 'running print checks',
  propose_concept: 'preparing concept',
  propose_character_design: 'creating character design',
  sculpt_character: 'molding 3D clay',
  update_mesh_prep: 'preparing print mesh',
  set_creation_mode: 'switching mode',
  list_concepts: 'checking designs',
  wait_for_concept: 'waiting for concept approval',
  generate_image: 'generating image',
  look_at_image: 'inspecting image',
  search_web: 'searching web',
  fetch_url: 'reading webpage',
}

export function AiPanel({ status, activity, images, sculpts = [], mlStatus, creationMode = 'engineer', directBuild = false, sculpting = false, context, concept, conceptOpen, onAnswerConcept, onBuildConcept, layout, zIndex, onFocusPanel, onOpenConcept, onOpenPhoto, onUploadImage, onDeleteImage, onSetupMl, onCancelMl, onUpdateSculptPrep, onSettings, onSend, onCancel, onReset, onRevert }: Props) {
  const mode = status?.mode ?? 'mcp'
  const pendingConcept = concept?.status === 'pending' && mode !== 'mcp' && Boolean(onAnswerConcept) ? concept : null
  const [text, setText] = useState('')
  const [rawTools, setRawTools] = useState<boolean>(() => {
    try { return localStorage.getItem('pa.rawTools') === '1' } catch { return false }
  })
  const toggleRawTools = (raw: boolean) => {
    setRawTools(raw)
    try { localStorage.setItem('pa.rawTools', raw ? '1' : '0') } catch { /* ignore */ }
  }
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
    const list = Array.from(files).filter((f) => f.type.startsWith('image/') || /\.(png|jpe?g|webp|heic|heif)$/i.test(f.name))
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
    if (pendingConcept && onAnswerConcept) {
      const note = attached.length
        ? `${t}\n\n(Reference image(s) attached: ${attached.join(', ')} — inspect with look_at_image.)`
        : t
      onAnswerConcept(pendingConcept.id, 'revise', note)
      setText('')
      setAttached([])
      return
    }
    const parts = [t]
    if (concept && (conceptOpen || concept.status === 'revise')) {
      parts.push(`(Active concept ${concept.id} "${concept.title}" [${concept.status}] — call look_at_concept(id="${concept.id}") to inspect its drawing.)`)
    }
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
        accept="image/png,image/jpeg,image/webp,image/heic,image/heif,.heic,.heif"
        multiple
        hidden
        onChange={(e) => { if (e.target.files) uploadFiles(e.target.files); e.target.value = '' }}
      />
      <button type="button" className="btn sm" disabled={uploading} onClick={() => imgInput.current?.click()}
        title="Upload a reference photo (PNG, JPG, WebP, HEIC) so the AI can inspect it">
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
              onDoubleClick={() => onOpenPhoto?.(img.id)}
              title={mode === 'mcp' ? `${img.id}: ${img.prompt || img.file} (double-click to view)` : `${on ? 'Detach' : 'Attach'} ${img.id}: ${img.prompt || img.file} (double-click to view)`}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 5px', border: 0, background: 'transparent' }}
            >
              <img src={api.imageUrl(img.id)} alt={img.id} style={{ width: 18, height: 18, objectFit: 'cover', borderRadius: 3 }} />
              <span>{img.id}</span>
            </button>
            <button
              type="button"
              aria-label={`View ${img.id}`}
              title={`View ${img.id}`}
              onClick={(e) => { e.stopPropagation(); onOpenPhoto?.(img.id) }}
              style={{ border: 0, background: 'transparent', color: 'var(--muted)', cursor: 'pointer', padding: '0 2px', display: 'inline-flex', alignItems: 'center' }}
            >
              <Icon name="eye" size={12} />
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
          <span className="spacer" />
          <div className="seg sm" role="group" aria-label="Chat output style">
            <button type="button" aria-pressed={!rawTools} onClick={() => toggleRawTools(false)}
              title="Natural-language view (hides raw tool calls)">Natural</button>
            <button type="button" aria-pressed={rawTools} onClick={() => toggleRawTools(true)}
              title="Show raw tool calls and JSON outputs">Raw tools</button>
          </div>
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
      <div className="row sm" style={{ padding: '6px 12px', gap: 8, borderBottom: '1px solid var(--line)', background: 'var(--panel)', alignItems: 'center', flexWrap: 'wrap' }}>
        <div className="seg sm" role="group" aria-label="Creation mode">
          <button type="button" aria-pressed={creationMode === 'engineer'}
            onClick={() => onSettings?.({ creation_mode: 'engineer' })}
            title="Engineer Mode: parametric CAD extrusions, sketches & exact dimensions">
            Engineer
          </button>
          <button type="button" aria-pressed={creationMode === 'character'}
            onClick={() => onSettings?.({ creation_mode: 'character' })}
            title="Character Mode: local 2D turnaround + organic 3D clay sculpting">
            Character
          </button>
        </div>
        <span className="spacer" />
        <label className="row sm" style={{ gap: 4, cursor: 'pointer', fontSize: 11.5 }}
          title="Bypass the Design File approval gate and build the 3D model directly from prompts">
          <input type="checkbox" checked={directBuild}
            onChange={(e) => onSettings?.({ direct_build: e.target.checked })} />
          <span>Direct build</span>
        </label>
      </div>
      {creationMode === 'character' && mlStatus && !mlStatus.ready && (
        <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--line)', background: 'color-mix(in srgb, var(--accent) 10%, var(--panel-2))', fontSize: 12 }}>
          <div className="row sm" style={{ gap: 8, alignItems: 'center' }}>
            <b>Local Character Engine (Free)</b>
            <span className="spacer" />
            <span className="nm" style={{ fontSize: 11 }}>{mlStatus.hardware.gpu_name} · {mlStatus.hardware.ram_gb} GB</span>
          </div>
          {mlStatus.installing ? (
            <div style={{ marginTop: 6 }}>
              <div className="row sm" style={{ gap: 6, marginBottom: 4 }}>
                <span>{mlStatus.stage || 'Installing…'} ({mlStatus.progress_pct}%)</span>
                <span className="spacer" />
                <button type="button" className="btn sm" onClick={onCancelMl}>Cancel</button>
              </div>
              <div style={{ height: 5, borderRadius: 3, background: 'var(--line)', overflow: 'hidden' }}>
                <div style={{ width: `${mlStatus.progress_pct}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.3s' }} />
              </div>
              {mlStatus.detail && <div className="nm" style={{ marginTop: 3, fontSize: 11 }}>{mlStatus.detail}</div>}
            </div>
          ) : (
            <div style={{ marginTop: 6 }}>
              <div className="nm" style={{ fontSize: 11.5, marginBottom: 6 }}>
                One-time free download (~{mlStatus.hardware.estimated_download_gb} GB) into your shared <code>ml-engine</code> folder ({mlStatus.hardware.sculptor_label}).
              </div>
              {mlStatus.error && <div style={{ color: 'var(--bad)', fontSize: 11.5, marginBottom: 6 }}>{mlStatus.error}</div>}
              <button type="button" className="btn primary sm" onClick={onSetupMl}>Set up Character Engine</button>
            </div>
          )}
        </div>
      )}
      {sculpting && (
        <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--line)', background: 'var(--panel-2)', fontSize: 12 }}>
          <b>Molding 3D clay &amp; repairing watertight mesh…</b>
        </div>
      )}
      {sculpts.length > 0 && onUpdateSculptPrep && (() => {
        const s = sculpts[sculpts.length - 1]
        return (
          <div style={{ padding: '6px 12px', borderBottom: '1px solid var(--line)', background: 'var(--panel-2)', fontSize: 11.5, display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
            <b>{s.id} Prep:</b>
            <label className="row sm" style={{ gap: 3 }}>
              <span>H</span>
              <input type="number" min={15} max={300} defaultValue={s.prep.target_height_mm} key={`${s.id}-h-${s.prep.target_height_mm}`}
                onBlur={(e) => { const v = Number(e.target.value); if (v > 0 && v !== s.prep.target_height_mm) onUpdateSculptPrep(s.id, { target_height_mm: v }) }}
                style={{ width: 52, fontSize: 11.5, padding: '1px 4px' }} />
              <span>mm</span>
            </label>
            <select aria-label="Sculpt display base" value={s.prep.base_style}
              onChange={(e) => onUpdateSculptPrep(s.id, { base_style: e.target.value as SculptPrep['base_style'] })}
              style={{ fontSize: 11.5, padding: '1px 4px' }}>
              <option value="none">No base</option>
              <option value="round">Round base</option>
              <option value="oval">Oval base</option>
              <option value="square">Square base</option>
            </select>
            <label className="row sm" style={{ gap: 3, cursor: 'pointer' }}>
              <input type="checkbox" checked={s.prep.keychain_loop}
                onChange={(e) => onUpdateSculptPrep(s.id, { keychain_loop: e.target.checked })} />
              <span>Loop</span>
            </label>
          </div>
        )
      })()}
      {context && <div className="ctx">Context: {context}</div>}
      {mode === 'mcp' && concept && !conceptOpen && (
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
            {(chat?.log ?? []).filter((e) => rawTools || e.role !== 'tool' || !e.ok).map((e, i) => (
              <div key={i} className={`entry ${e.role}${e.ok ? '' : ' error'}`}>
                {e.role === 'tool'
                  ? <><span className="who">{e.tool}</span><div className="toolout">{e.text}</div></>
                  : e.role === 'assistant'
                    ? <div className="md" dangerouslySetInnerHTML={{ __html: renderMarkdown(e.text) }} />
                    : <div>{e.text}</div>}
              </div>
            ))}
            {concept && (
              <div className="entry assistant" style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span style={{ fontWeight: 600, fontSize: 12.5 }}>{concept.id} · {concept.title}</span>
                <span className={`chip cstat ${concept.status}`}>
                  {concept.status === 'pending' ? 'Waiting for you' : concept.status === 'approved' ? 'Approved' : 'Changes asked'}
                </span>
                <span className="spacer" />
                <button className="btn sm" onClick={() => onOpenConcept(concept.id)}>
                  {conceptOpen ? 'Viewing design' : 'View design'}
                </button>
                {concept.status === 'pending' && onAnswerConcept ? (
                  <button className="btn primary sm" onClick={() => { onAnswerConcept(concept.id, 'approve', text.trim()); setText('') }}>
                    Approve & build
                  </button>
                ) : onBuildConcept ? (
                  <button className="btn primary sm" disabled={chat?.running} onClick={() => { onBuildConcept(concept, text.trim()); setText('') }}>
                    {concept.status === 'approved' ? 'Build' : 'Approve & build'}
                  </button>
                ) : null}
              </div>
            )}
            {chat?.running && (() => {
              const toolEntries = (chat.log ?? []).filter((e) => e.role === 'tool')
              const lastEntry = toolEntries.at(-1)
              const activeKey = chat.current_step || lastEntry?.tool || null
              const waitingOnUser = activeKey === 'wait_for_concept' || Boolean(pendingConcept && !chat.current_step)
              const activeLabel = waitingOnUser
                ? `Waiting for your approval${pendingConcept ? ` on ${pendingConcept.id}` : ''}`
                : (activeKey ? (TOOL_LABELS[activeKey] ?? activeKey) : 'Thinking')
              const elapsed = chat.elapsed_s ?? 0
              const idle = chat.idle_s ?? 0
              const steps = chat.steps ?? toolEntries.length
              const fmtTime = (s: number) => s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`
              const stalled = !waitingOnUser && idle >= 30
              return (
                <div className={`entry assistant working${stalled ? ' stalled' : ''}`}>
                  <div className="work-row">
                    <span className={`work-dot${stalled ? ' stalled' : ''}`} aria-hidden="true" />
                    <b>{waitingOnUser ? activeLabel : `${activeLabel}…`}</b>
                    <span className="work-meta">
                      {steps > 0 ? `Step ${steps} · ` : ''}{fmtTime(elapsed)}
                      {waitingOnUser
                        ? ' · click Approve & build or reply below'
                        : (idle >= 5 ? ` · ${stalled ? `no activity ${fmtTime(idle)}` : `active ${fmtTime(idle)} ago`}` : ' · active')}
                      {(chat.queued ?? 0) > 0 ? ` · ${chat.queued} queued` : ''}
                    </span>
                  </div>
                  {lastEntry && !rawTools && (
                    <div className="work-last">
                      {lastEntry.ok ? '✓' : '✗'} {TOOL_LABELS[lastEntry.tool ?? ''] ?? lastEntry.tool}
                      {lastEntry.text ? `: ${lastEntry.text.replace(/\s+/g, ' ').slice(0, 90)}` : ''}
                    </div>
                  )}
                </div>
              )
            })()}
            <div ref={end} />
          </div>
          {refStrip}
          <div className="aifoot col">
            <label htmlFor="ai-msg" className="nm">
              {pendingConcept ? `Reply to concept ${pendingConcept.id} (or click Approve & build above)` : 'Ask for a change'}
            </label>
            <div className="row">
              <textarea id="ai-msg" rows={3} value={text} onChange={(e) => setText(e.target.value)}
                placeholder={pendingConcept
                  ? 'Ask for changes to this concept, or click Approve & build above…'
                  : chat?.running
                    ? 'Send follow-up instructions or attach photos while the AI works…'
                    : 'e.g. make the walls 2 mm and twist it 90° (paste or + Photo to attach reference images)'}
                onPaste={(e) => {
                  const files = e.clipboardData?.files
                  if (files?.length && Array.from(files).some((f) => f.type.startsWith('image/'))) {
                    e.preventDefault()
                    uploadFiles(files)
                  }
                }}
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send() }} />
              {chat?.running ? (
                <div className="col" style={{ gap: 4 }}>
                  <button className="send" aria-label="Send" onClick={send} disabled={!text.trim()}
                    title={pendingConcept ? 'Send feedback on this concept' : 'Send follow-up instructions (queued if AI is busy)'}>
                    <Icon name="send" size={18} width={2} />
                  </button>
                  <button className="btn sm" onClick={onCancel}>Stop</button>
                </div>
              ) : (
                <button className="send" aria-label="Send" onClick={send} disabled={!text.trim()}><Icon name="send" size={18} width={2} /></button>
              )}
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
