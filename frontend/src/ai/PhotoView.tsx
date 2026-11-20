// Reference & generated photo viewer shown over the 3D canvas via the Photos tab or thumbnail double-click.
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, type ImageInfo } from '../api'

interface Props {
  image: ImageInfo
  images: ImageInfo[]
  onSelectImage: (id: string) => void
  onClose: () => void
}

const labelFor = (img: ImageInfo) => (img.prompt && img.prompt !== 'uploaded reference' ? img.prompt : img.file)

export function PhotoView({ image: img, images, onSelectImage, onClose }: Props) {
  const [expanded, setExpanded] = useState(false)
  const idx = images.findIndex((x) => x.id === img.id)
  const hasPrev = idx > 0
  const hasNext = idx >= 0 && idx < images.length - 1

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopPropagation()
        if (expanded) setExpanded(false)
        else onClose()
      } else if (e.key === 'ArrowLeft' && hasPrev) {
        onSelectImage(images[idx - 1].id)
      } else if (e.key === 'ArrowRight' && hasNext) {
        onSelectImage(images[idx + 1].id)
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [expanded, hasPrev, hasNext, idx, images, onSelectImage, onClose])

  const src = api.imageUrl(img.id)
  const title = labelFor(img)
  const fullScreen = () => setExpanded((v) => !v)

  return (
    <section className="concept photo-view" role="dialog" aria-label={`Photo ${img.id}: ${title}`}>
      <div className="dhead">
        {images.length > 1 ? (
          <div className="row sm" style={{ gap: 6, minWidth: 0 }}>
            <select
              aria-label="Select photo"
              value={img.id}
              onChange={(e) => onSelectImage(e.target.value)}
              style={{ fontSize: 13, fontWeight: 600, padding: '2px 6px', maxWidth: 420 }}
            >
              {images.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.id} · {labelFor(item)}
                </option>
              ))}
            </select>
            <button type="button" className="btn sm" disabled={!hasPrev}
              aria-label="Previous photo" title="Previous photo (←)"
              onClick={() => hasPrev && onSelectImage(images[idx - 1].id)}>
              ‹
            </button>
            <button type="button" className="btn sm" disabled={!hasNext}
              aria-label="Next photo" title="Next photo (→)"
              onClick={() => hasNext && onSelectImage(images[idx + 1].id)}>
              ›
            </button>
          </div>
        ) : (
          <h2>{img.id} · {title}</h2>
        )}
        <span className="chip">{img.actor === 'user' ? 'Uploaded' : 'Generated'}</span>
        <span className="spacer" />
        <button type="button" className="btn" onClick={fullScreen} title="Show photo full screen (Esc to come back; or double-click the photo)">
          ⛶ Full screen
        </button>
        <button type="button" className="x" onClick={onClose} aria-label="Close" title="Close">×</button>
      </div>
      <div className="cbody">
        <div className="cdraw photo-stage" onDoubleClick={fullScreen} title="Double-click for full screen">
          <img src={src} alt={`Reference photo ${img.id}: ${title}`} />
        </div>
        <div className="photo-meta">
          <span><b>{img.id}</b> · {img.file}</span>
          {img.prompt && img.prompt !== 'uploaded reference' ? <span className="dim"> — {img.prompt}</span> : null}
          <span className="spacer" />
          <span className="dim">{new Date(img.created * 1000).toLocaleString()}</span>
        </div>
      </div>
      {expanded && createPortal(
        <div className="cdraw-fullscreen photo-fullscreen" role="dialog" aria-label={`Full screen photo: ${title}`} onDoubleClick={() => setExpanded(false)}>
          <button type="button" className="btn cdraw-exit" onClick={() => setExpanded(false)} title="Exit full screen (Esc)">
            ✕ Exit full screen
          </button>
          <img src={src} alt={`Reference photo ${img.id}: ${title}`} />
        </div>,
        document.body,
      )}
    </section>
  )
}
