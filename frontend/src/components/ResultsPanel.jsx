import { Line, Area, AreaChart, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, BarChart, Bar, CartesianGrid, ComposedChart } from "recharts";
import { Activity, Hash, TreeDeciduous, Droplets, Gauge, Sparkles, Shield, AlertCircle } from "lucide-react";

const PRIORITY_STYLE = {
  low:      { bg: "bg-[var(--leaf-50)]",  text: "text-[var(--leaf-800)]", ring: "ring-[var(--leaf-300)]" },
  moderate: { bg: "bg-amber-50",          text: "text-amber-900",         ring: "ring-amber-200" },
  high:     { bg: "bg-orange-50",         text: "text-orange-900",        ring: "ring-orange-200" },
  critical: { bg: "bg-red-50",            text: "text-red-900",           ring: "ring-red-300" },
};

export default function ResultsPanel({ result }) {
  const r = result;
  const forecast = r.carbon_scenario_bau.map((v, i) => ({
    year: `Y${i + 1}`,
    BAU: v,
    Mitigation: r.carbon_scenario_mitigation[i],
    Unchecked: r.carbon_scenario_unchecked?.[i] ?? v,
    ci_lo: r.carbon_ci_lower[i],
    ci_hi: r.carbon_ci_upper[i],
  }));
  const cumulative = (r.carbon_cumulative_bau || []).map((v, i) => ({
    year: `Y${i + 1}`,
    "If not followed": r.carbon_cumulative_unchecked[i],
    "Business-as-usual": v,
    "If followed": r.carbon_cumulative_mitigation[i],
  }));
  const classNames = r.recommendation.class_names.map((n) => n.replace(/_/g, " "));
  const stds = r.recommendation.class_probabilities_std || [];
  const probs = r.recommendation.class_probabilities.map((p, i) => ({
    class: classNames[i],
    probability: p,
    upper: Math.min(1, p + (stds[i] || 0)),
    lower: Math.max(0, p - (stds[i] || 0)),
  }));
  const priority = PRIORITY_STYLE[r.recommendation.priority] || PRIORITY_STYLE.low;
  const epi = r.recommendation.epistemic_uncertainty || 0;

  return (
    <section data-testid="results-panel" className="mt-12 reveal">
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <Stat icon={Gauge}   label="AVLE+ / km²"   value={r.avle_plus_per_km2.toFixed(3)} sub={`raw ${r.avle_plus.toFixed(1)}`} />
        <Stat icon={Activity} label="Carbon (tCO₂)" value={r.carbon_estimate_tco2.toFixed(1)} sub={`biome · ${r.biome.replace(/_/g," ")}`} />
        <Stat icon={TreeDeciduous} label="Loss area (ha)" value={r.loss_area_ha.toFixed(2)} sub={`${r.loss_pixel_count} px @ 10 m`} />
        <Stat icon={Droplets} label="ΔNDVI (norm.)" value={r.ndvi_delta_mean.toFixed(3)} sub={`z-scored · t₁ ${r.ndvi_t1_mean.toFixed(2)} → t₂ ${r.ndvi_t2_mean.toFixed(2)}`} />
      </div>

      <div className="grid lg:grid-cols-3 gap-4 mb-6">
        <ImgCard title="NDVI · baseline"    src={r.images.ndvi_t1} testid="img-ndvi-t1"/>
        <ImgCard title="NDVI · observation" src={r.images.ndvi_t2} testid="img-ndvi-t2"/>
        <ImgCard title="Loss overlay (red = confirmed, orange = filtered)" src={r.images.loss_overlay} testid="img-loss"/>
      </div>

      {/* ---------------- FUTURE SCENARIOS ---------------- */}
      <div className="grid lg:grid-cols-5 gap-4 mb-6">
        <div className="lg:col-span-3 leaf-card rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-lg" style={{fontFamily:"var(--font-display)"}}>Annual emissions · 5-year outlook</h4>
              <p className="text-xs text-[var(--stone)] mt-0.5">XGBoost · 95 % quantile CI · intervention factor {r.intervention_factor?.toFixed(2)}</p>
            </div>
          </div>
          <div style={{width:"100%", height: 260}}>
            <ResponsiveContainer>
              <AreaChart data={forecast} margin={{top:4,right:10,left:-12,bottom:0}}>
                <CartesianGrid stroke="#e6efdc" vertical={false}/>
                <XAxis dataKey="year" stroke="#6b7560" tick={{fontSize:12}}/>
                <YAxis stroke="#6b7560" tick={{fontSize:12}}/>
                <Tooltip contentStyle={{borderRadius:10, border:"1px solid #d7e7ca"}}/>
                <Legend wrapperStyle={{fontSize:12}}/>
                <Area type="monotone" dataKey="ci_hi" stroke="none" fill="#c7e3b6" fillOpacity={0.35} name="CI upper"/>
                <Area type="monotone" dataKey="ci_lo" stroke="none" fill="#ffffff" fillOpacity={1} name="CI lower"/>
                <Line type="monotone" dataKey="Unchecked"  stroke="#a03a1a" strokeDasharray="4 4" strokeWidth={2} dot={{r:3}}/>
                <Line type="monotone" dataKey="BAU"        stroke="#b2522a" strokeWidth={2.5} dot={{r:3}}/>
                <Line type="monotone" dataKey="Mitigation" stroke="#2f6222" strokeWidth={2.5} dot={{r:3}}/>
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="lg:col-span-2 leaf-card rounded-2xl p-6">
          <h4 className="text-lg mb-1" style={{fontFamily:"var(--font-display)"}}>Cumulative 5-yr emissions</h4>
          <p className="text-xs text-[var(--stone)] mb-3">If followed vs if not followed (total tCO₂)</p>
          <div style={{width:"100%", height: 220}}>
            <ResponsiveContainer>
              <ComposedChart data={cumulative} margin={{top:4,right:10,left:-12,bottom:0}}>
                <CartesianGrid stroke="#e6efdc" vertical={false}/>
                <XAxis dataKey="year" stroke="#6b7560" tick={{fontSize:12}}/>
                <YAxis stroke="#6b7560" tick={{fontSize:12}}/>
                <Tooltip contentStyle={{borderRadius:10, border:"1px solid #d7e7ca"}}
                         formatter={(v)=>Number(v).toFixed(0)}/>
                <Legend wrapperStyle={{fontSize:11}}/>
                <Area type="monotone" dataKey="If not followed" stroke="#a03a1a" fill="#fcd9cf" fillOpacity={0.6}/>
                <Area type="monotone" dataKey="Business-as-usual" stroke="#b2522a" fill="#f9e3cc" fillOpacity={0.45}/>
                <Area type="monotone" dataKey="If followed" stroke="#2f6222" fill="#c7e3b6" fillOpacity={0.55}/>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 text-xs text-[var(--leaf-900)]">
            Savings over 5 yrs (followed vs BAU): <strong className="mono-num">{((r.carbon_cumulative_bau.at(-1) - r.carbon_cumulative_mitigation.at(-1))).toFixed(0)} tCO₂</strong>
          </div>
        </div>
      </div>

      {/* ---------------- RECOMMENDATION + UNCERTAINTY ---------------- */}
      <div className={`rounded-2xl p-6 mb-6 ring-1 ${priority.ring} ${priority.bg}`}>
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider opacity-70">
            <Sparkles size={14}/> Neural recommendation · {r.recommendation.mc_passes} Monte-Carlo dropout passes
          </div>
          <div className="flex items-center gap-2 text-xs">
            <AlertCircle size={14} className={priority.text}/>
            <span className={priority.text}>Epistemic uncertainty · {(epi * 100).toFixed(1)} %</span>
          </div>
        </div>
        <h4 data-testid="rec-action" className={`mt-2 text-xl ${priority.text}`} style={{fontFamily:"var(--font-display)"}}>
          {r.recommendation.action}
        </h4>
        <div className={`mt-1 text-sm ${priority.text} opacity-80`}>
          priority · <span className="font-semibold">{r.recommendation.priority}</span>
          &nbsp;·&nbsp; mean confidence · <span className="mono-num">{(r.recommendation.confidence*100).toFixed(1)} %</span>
          &nbsp;·&nbsp; predictive entropy · <span className="mono-num">{(r.recommendation.entropy_normalised || 0).toFixed(2)}</span>
        </div>
        <div className="mt-5 bg-white/50 rounded-xl p-3" style={{width:"100%", height: 200}}>
          <ResponsiveContainer>
            <BarChart data={probs} layout="vertical" margin={{top:0,right:10,left:4,bottom:0}}>
              <XAxis type="number" domain={[0,1]} hide/>
              <YAxis dataKey="class" type="category" width={160} tick={{fontSize:11, fill:"#3b4a35"}}/>
              <Tooltip formatter={(v, n)=>[`${(v*100).toFixed(1)} %`, n]}/>
              <Bar dataKey="probability" fill="#4f9938" radius={[0,6,6,0]}/>
              <Bar dataKey="upper" fill="none" stroke="#2f6222" strokeWidth={1}/>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] mt-2 text-[var(--stone)]">
          Probabilities are posterior means across {r.recommendation.mc_passes} forward passes with dropout active; bar thickness = mean, std across passes controls the implicit error bar.
        </p>
      </div>

      {/* ---------------- PROVENANCE ---------------- */}
      <div className="leaf-card rounded-2xl p-6 grid md:grid-cols-2 gap-6">
        <div>
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-[var(--leaf-700)] mb-2">
            <Hash size={14}/> Region hash
          </div>
          <div data-testid="region-hash" className="font-mono text-sm break-all text-[var(--leaf-900)]">{r.region_hash}</div>

          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-[var(--leaf-700)] mt-4 mb-2">
            <Shield size={14}/> Blockchain receipt
          </div>
          {r.blockchain ? (
            <div className="space-y-1 text-sm">
              <div data-testid="chain-tx"      className="font-mono break-all">tx · {r.blockchain.tx_hash}</div>
              <div className="text-[var(--stone)] mono-num">block {r.blockchain.block_number} · gas {r.blockchain.gas_used} · chain_id {r.blockchain.chain_id}</div>
            </div>
          ) : <div className="text-sm text-[var(--stone)]">not logged</div>}
        </div>
        <div>
          <div className="text-xs uppercase tracking-wider text-[var(--leaf-700)] mb-2">Provenance</div>
          <dl className="text-sm grid grid-cols-2 gap-y-1">
            <dt className="text-[var(--stone)]">t₁ source</dt><dd>{r.source.t1}</dd>
            <dt className="text-[var(--stone)]">t₂ source</dt><dd>{r.source.t2}</dd>
            <dt className="text-[var(--stone)]">biome</dt><dd>{r.biome.replace(/_/g," ")}</dd>
            <dt className="text-[var(--stone)]">NDVI normalised</dt><dd>{r.ndvi_normalised ? "yes (z-score)" : "no"}</dd>
            <dt className="text-[var(--stone)]">SSS</dt><dd className="mono-num">{(r.scenario_separation_score || 0).toFixed(3)}</dd>
          </dl>
        </div>
      </div>
    </section>
  );
}

function Stat({ icon: Icon, label, value, sub }) {
  return (
    <div className="leaf-card rounded-2xl p-5">
      <Icon className="text-[var(--leaf-700)]" size={18}/>
      <div className="text-xs uppercase tracking-wider text-[var(--stone)] mt-3">{label}</div>
      <div className="text-3xl mono-num mt-1" style={{fontFamily:"var(--font-display)"}}>{value}</div>
      {sub && <div className="text-xs text-[var(--stone)] mt-1">{sub}</div>}
    </div>
  );
}

function ImgCard({ title, src, testid }) {
  return (
    <div className="leaf-card rounded-2xl p-3">
      <div className="px-2 pt-1 pb-2 text-xs uppercase tracking-wider text-[var(--stone)]">{title}</div>
      <img data-testid={testid} src={src} alt={title} className="rounded-lg w-full aspect-square object-cover"/>
    </div>
  );
}
