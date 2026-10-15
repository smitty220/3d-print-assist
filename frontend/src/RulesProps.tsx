import type { Rules } from './api'
import { NumField } from './fields'
import type { Units } from './units'

interface Props {
  rules: Rules
  materials: Record<string, number>
  units: Units
  onChange: (rules: Rules) => void
}

const LB_PER_IN3_PER_KG_M3 = 0.0000361273

export function RulesProps({ rules, materials, units, onChange }: Props) {
  const set = (patch: Partial<Rules>) => onChange({ ...rules, ...patch })
  const density = materials[rules.material]
  const num = (key: keyof Rules, label: string, quantity: 'length' | 'ratio' | 'mass' | 'stress', extra?: { allowEmpty?: boolean }) => (
    <NumField id={`rule-${key}`} label={label} value={rules[key] as number | null} quantity={quantity} units={units} min={0}
      allowEmpty={extra?.allowEmpty} onCommit={(v) => set({ [key]: v } as Partial<Rules>)} />
  )

  return (
    <>
      <div className="ttl">
        <b>Design rules</b>
        <span>project defaults</span>
      </div>
      <div className="field">
        <label htmlFor="rule-material">Material</label>
        <select id="rule-material" value={rules.material} onChange={(e) => set({ material: e.target.value })}>
          {Object.keys(materials).map((m) => <option key={m}>{m}</option>)}
        </select>
      </div>
      {density != null && (
        <dl className="kv">
          <dt>Density</dt>
          <dd>{units === 'in' ? `${(density * LB_PER_IN3_PER_KG_M3).toFixed(4)} lb/in³` : `${density.toFixed(0)} kg/m³`}</dd>
        </dl>
      )}
      <div className="sec">Walls and floors</div>
      <div className="row2">
        {num('min_wall', 'Min wall', 'length')}
        {num('min_wall_preferred', 'Preferred wall', 'length')}
      </div>
      <div className="row2">
        {num('min_floor', 'Min pocket floor', 'length')}
        {num('wall_height_thickness_max', 'Max wall h/t', 'ratio')}
      </div>
      <div className="sec">Fasteners and lugs (e/D, t/D)</div>
      <div className="row2">
        {num('bolt_edge_min', 'Bolt e/D min', 'ratio')}
        {num('bolt_edge_preferred', 'Bolt e/D preferred', 'ratio')}
      </div>
      <div className="row2">
        {num('lug_edge_full_bearing', 'Lug e/D full bearing', 'ratio')}
        {num('lug_edge_min', 'Lug e/D min', 'ratio')}
      </div>
      <div className="row2">
        {num('lug_td_min', 'Lug t/D min', 'ratio')}
        {num('pocket_depth_tool_max', 'Pocket depth / tool Ø', 'ratio')}
      </div>
      <div className="sec">Allowables (stress analysis)</div>
      <div className="row2">
        {num('ftu', 'Ftu override', 'stress', { allowEmpty: true })}
        {num('fty', 'Fty override', 'stress', { allowEmpty: true })}
      </div>
      <p className="note">Empty uses the material's MIL-HDBK-5J A-basis LT values (shown in the stress results). Enter your own allowables
        to override them, e.g. from your company data.</p>
      <div className="sec">Targets (soft)</div>
      {num('mass_target', 'Mass target', 'mass', { allowEmpty: true })}
      <p className="note">Seeded from the researched rule table in the plan doc; adjust to your company standards. Leave the mass target empty for none.</p>
    </>
  )
}
