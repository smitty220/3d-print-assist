// Interim shell while the backend is rebuilt (milestone M1.0): the viewer, the feature list and the feature panel.
// The full layout (tree, toolbars, AI panel) replaces this in M1.8.
import { useEffect, useRef, useState } from 'react'
import { api, type DesignState, type FeaturePatch, type FeatureSpec } from './api'
import { FeatureProps } from './FeatureProps'
import { Viewer } from './viewer/Viewer'

const DESIGN = '__part__'
const DESIGN_STYLE = { color: 0xb7bfc8, opacity: 1, cap: { fill: 0xe3c79a, line: 0x6b4d1e } }

export default function App() {
  const vpRef = useRef<HTMLDivElement>(null)
  const viewer = useRef<Viewer | null>(null)
  const [design, setDesign] = useState<DesignState | null>(null)
  const [specs, setSpecs] = useState<FeatureSpec[]>([])
  const [sel, setSel] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const v = new Viewer(vpRef.current!)
    viewer.current = v
    return () => {
      v.dispose()
      viewer.current = null
    }
  }, [])

  const show = (d: DesignState, fit = false) => {
    setDesign(d)
    const v = viewer.current
    if (!v) return
    if (d.mesh) v.addPart({ ...d.mesh, name: DESIGN }, DESIGN_STYLE)
    else v.removePart(DESIGN)
    if (fit) v.fit()
  }

  useEffect(() => {
    Promise.all([api.featureCatalog(), api.design()])
      .then(([c, d]) => {
        setSpecs(c)
        show(d, true)
      })
      .catch((e) => setError(String(e)))
  }, [])

  const patch = (id: string, p: FeaturePatch) => api.patchFeature(id, p).then((d) => show(d)).catch((e) => setError(String(e)))
  const feature = design?.features.find((f) => f.id === sel)
  const spec = specs.find((s) => s.kind === feature?.kind)

  return (
    <div className="app">
      <header className="top"><b>3D Print Assist</b></header>
      <main>
        <aside className="side">
          <div className="sec">Features</div>
          {design?.features.map((f) => (
            <button key={f.id} className={'row' + (f.id === sel ? ' sel' : '')} onClick={() => setSel(f.id)}>
              {f.id} {f.kind}{f.error ? ' ✖' : ''}
            </button>
          ))}
        </aside>
        <section className="vpwrap" ref={vpRef} />
        <aside className="side">
          {feature && spec && (
            <FeatureProps feature={feature} spec={spec} checks={design?.checks.filter((c) => c.feature === feature.id) ?? []}
              objects={[]} features={design?.features ?? []} faceGroups={[]} units="mm"
              onPatch={(p) => patch(feature.id, p)} onDelete={() => api.deleteFeature(feature.id).then((d) => show(d))}
              onEditSketch={() => {}} pickingEdges={false} onPickEdges={() => {}} />
          )}
        </aside>
      </main>
      <footer>{error}</footer>
    </div>
  )
}
