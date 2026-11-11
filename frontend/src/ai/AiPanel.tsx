// The AI panel (always on the right). Two modes: Claude Code drives the app over MCP (no key needed; this panel shows
// what it does, live), or the in-app chat calls the Claude API (for the home server).
import { useEffect, useRef, useState } from 'react'
import type { Activity, AiStatus, Concept } from '../api'
import { Icon } from '../ui/icons'

interface Props {
  status: AiStatus | null
  activity: Activity[]
  context: string | null
  concept: Concept | null // the newest
  onOpenConcept: (id: string) => void
  onMode: (m: 'mcp' | 'api') => void
  onSettings?: (patch: { ai_provider?: 'auto' | 'gemini' | 'anthropic'; ai_model?: string }) => void
  onSend: (text: string) => void
  onCancel: () => void
  onReset: () => void
  onRevert: () => void
}

const time = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

export function AiPanel({ status, activity, context, concept, onOpenConcept, onMode, onSettings, onSend, onCancel, onReset, onRevert }: Props) {
  const mode = status?.mode ?? 'mcp'
  const [text, setText] = useState('')
  const end = useRef<HTMLDivElement>(null)
  const chat = status?.chat
  useEffect(() => { end.current?.scrollIntoView({ block: 'end' }) }, [chat?.log.length, activity.length])
  // The AI's changes, and the user's answers to its concepts (the conversation about what to build).
  const ai = activity.filter((a) => a.actor !== 'user' || a.text.startsWith('Concept '))
  const send = () => {
    const t = text.trim()
    if (!t) return
    onSend(context ? `${t}\n\n(The user has ${context} selected.)` : t)
    setText('')
  }

  return (
    <aside className="side ai" aria-label="AI assistant">
      <div className="aihead">
        <h2>AI assistant</h2>
        <span className="spacer" />
        <div className="seg sm" role="group" aria-label="AI mode">
          <button aria-pressed={mode === 'mcp'} onClick={() => onMode('mcp')} title="Claude Code or MCP agent connects from terminal">MCP Agent</button>
          <button aria-pressed={mode === 'api'} onClick={() => onMode('api')} title="In-App AI Chat (Gemini / Claude)">In-App Chat</button>
        </div>
      </div>
      {mode === 'api' && <div className="row sm" style={{ padding: '6px 12px', gap: 8, borderBottom: '1px solid var(--line)', background: 'var(--panel-2)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: 2 }}>
          <label htmlFor="ai-provider-select" style={{ fontSize: 10, textTransform: 'uppercase', opacity: 0.65, letterSpacing: '0.04em' }}>Provider</label>
          <select
            id="ai-provider-select"
            value={status?.settings_provider || 'auto'}
            onChange={(e) => {
              onSettings?.({ ai_provider: e.target.value as 'auto' | 'gemini' | 'anthropic', ai_model: '' })
            }}
            style={{ fontSize: 12, padding: '3px 6px' }}
          >
            <option value="auto">Auto ({status?.provider === 'gemini' ? 'Gemini' : 'Claude'})</option>
            <option value="gemini">Google Gemini</option>
            <option value="anthropic">Anthropic Claude</option>
          </select>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1.5, gap: 2 }}>
          <label htmlFor="ai-model-select" style={{ fontSize: 10, textTransform: 'uppercase', opacity: 0.65, letterSpacing: '0.04em' }}>Model</label>
          <select
            id="ai-model-select"
            value={status?.settings_model || status?.model || ''}
            onChange={(e) => {
              onSettings?.({ ai_model: e.target.value })
            }}
            style={{ fontSize: 12, padding: '3px 6px' }}
          >
            {(status?.available_models?.[status?.provider || 'gemini'] || []).map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </div>
      </div>}
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

      {mode === 'mcp' ? (
        <>
          <div className="log">
            {ai.length === 0 && (
              <div className="intro">
                <p>Run Claude Code or your MCP client in the <code>3d-print-assist</code> folder while this app is open. It connects through
                  the <code>3d-print-assist</code> MCP server and designs with the same tools you have; you watch each change
                  land here and in the tree, and can edit alongside it.</p>
                <p>To chat directly with <b>{status?.provider === 'gemini' ? 'Google Gemini' : 'Claude'}</b> in this window, switch to <b>In-App Chat</b> above.</p>
              </div>
            )}
            {ai.map((a, i) => (
              <div key={i} className={`entry ${a.ok ? 'tool' : 'error'}`}>
                <span className="who">{a.actor}</span><span className="when">{time(a.at)}</span>
                <div>{a.text}</div>
              </div>
            ))}
            <div ref={end} />
          </div>
          <div className="aifoot">
            <button className="btn" onClick={onRevert} disabled={!ai.some((a) => a.ok && a.actor !== 'user')}>Revert last AI change</button>
          </div>
        </>
      ) : (
        <>
          <div className="log">
            {!status?.key && (
              <div className="intro">
                <p>
                  No {status?.provider === 'gemini' ? 'Gemini' : 'Claude'} API key yet. Put{' '}
                  <code>{status?.provider === 'gemini' ? 'GEMINI_API_KEY=…' : 'ANTHROPIC_API_KEY=…'}</code> (or{' '}
                  <code>{status?.provider === 'gemini' ? 'ANTHROPIC_API_KEY=…' : 'GEMINI_API_KEY=…'}</code>) in{' '}
                  <code>backend/.env</code>; it is read on the next message, no restart needed. Model: {status?.model ?? '…'}.
                </p>
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
          <div className="aifoot col">
            <label htmlFor="ai-msg" className="nm">Ask for a change</label>
            <div className="row">
              <textarea id="ai-msg" rows={3} value={text} onChange={(e) => setText(e.target.value)} disabled={chat?.running}
                placeholder="e.g. make the walls 2 mm and twist it 90°"
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
    </aside>
  )
}
