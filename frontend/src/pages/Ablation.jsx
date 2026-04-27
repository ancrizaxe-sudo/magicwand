import { useEffect, useState } from "react";
import { getAblation } from "../lib/api";
import { Loader2, Trophy, FlaskConical, GitCompare } from "lucide-react";

const COLS = [
  { k: "rmse",     label: "RMSE (tCO₂)", fmt: (v) => v.toFixed(1) },
  { k: "mae",      label: "MAE",         fmt: (v) => v.toFixed(1) },
  { k: "r2",       label: "R²",          fmt: (v) => v.toFixed(3) },
  { k: "spearman", label: "Spearman ρ²", fmt: (v) => v.toFixed(3) },
];

const FAMILY_BADGE = {
  "baseline":          "bg-stone-100 text-stone-700",
  "strong baseline":   "bg-amber-50 text-amber-800",
  "avle progression":  "bg-[var(--leaf-50)] text-[var(--leaf-800)]",
  "full system":       "bg-[var(--leaf-100)] text-[var(--leaf-800)]",
  "segmentation":      "bg-blue-50 text-blue-800",
};

export default function Ablation() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [regime, setRegime] = useState("structured");
  useEffect(() => { getAblation().then(setData).catch((e) => setErr(e.message)); }, []);

  if (err) return <Center msg={`Ablation not available: ${err}`}/>;
  if (!data) return <Center msg={<><Loader2 className="animate-spin inline mr-2"/>Loading …</>}/>;

  const rows = regime === "structured" ? data.structured_ablation : data.neutral_ablation;
  const bestIdx = rows.length - 1;
  const seg = data.segmentation || {};

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <div className="mb-10 reveal">
        <div className="text-xs uppercase tracking-[0.22em] text-[var(--leaf-700)] mb-2">table · 1 · progressive ablation</div>
        <h1 className="text-4xl sm:text-5xl" style={{fontFamily:"var(--font-display)"}}>Two-regime ablation</h1>
        <p className="mt-3 text-[#3b4a35] max-w-3xl">
          Every AVLE+ component is added progressively. Strong baselines (NDVI + smoothing, + clustering, + temporal averaging) are included
          so the improvement isn't measured against a straw-man. Each configuration is evaluated in two synthetic regimes — <em>structured</em>
          and <em>neutral</em> — to detect circular validation.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 mb-6 reveal reveal-d1">
        <button data-testid="ablation-regime-structured" onClick={() => setRegime("structured")}
                className={regime === "structured" ? "pill-btn" : "pill-ghost"}>
          <FlaskConical size={14}/> Structured regime
        </button>
        <button data-testid="ablation-regime-neutral" onClick={() => setRegime("neutral")}
                className={regime === "neutral" ? "pill-btn" : "pill-ghost"}>
          <GitCompare size={14}/> Neutral (null) regime
        </button>
        <div className="ml-auto text-xs text-[var(--stone)] self-center max-w-md text-right">
          {regime === "structured"
            ? "AVLE+ expected to progressively improve over baselines."
            : "AVLE+ should NOT show gains here — this is the null-hypothesis regime."}
        </div>
      </div>

      <div className="leaf-card rounded-2xl overflow-hidden reveal reveal-d1">
        <table className="w-full text-sm" data-testid="ablation-table">
          <thead className="bg-[var(--leaf-50)] text-[var(--leaf-800)]">
            <tr>
              <th className="text-left p-4 font-semibold">Method</th>
              <th className="text-left p-4 font-semibold">Family</th>
              {COLS.map(({k, label}) => (
                <th key={k} className="text-right p-4 font-semibold">{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className={`${i===bestIdx ? "bg-[var(--leaf-100)]/60 font-semibold" : ""} border-t border-[#e6efdc]`}>
                <td className="p-4 text-[var(--leaf-900)]">
                  <div className="flex items-center gap-2">
                    {i===bestIdx && <Trophy size={14} className="text-[var(--leaf-700)]"/>}
                    {row.method}
                  </div>
                </td>
                <td className="p-4">
                  <span className={`inline-block text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full ${FAMILY_BADGE[row.family] || "bg-stone-100 text-stone-700"}`}>
                    {row.family}
                  </span>
                </td>
                {COLS.map(({k, fmt}) => (
                  <td key={k} className="p-4 text-right mono-num">
                    {row[k] == null ? "—" : fmt(row[k])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Segmentation ablation */}
      <div className="grid md:grid-cols-2 gap-4 mt-10">
        <div className="leaf-card rounded-2xl p-6">
          <h4 className="text-lg mb-3" style={{fontFamily:"var(--font-display)"}}>Segmentation · IoU</h4>
          <p className="text-xs text-[var(--stone)] mb-4">Over {seg.n} matched test patches. U-Net is trained on ESA WorldCover + Sentinel-2.</p>
          <Row label="NDVI threshold (baseline)" value={seg.ndvi_iou?.toFixed(3) ?? "—"}/>
          <Row label="U-Net (trained)"           value={seg.unet_iou?.toFixed(3) ?? "—"}/>
          <Row label="Δ (U-Net − NDVI)"          value={seg.delta_iou != null ? seg.delta_iou.toFixed(3) : "—"}/>
        </div>

        <div className="leaf-card rounded-2xl p-6">
          <h4 className="text-lg mb-3" style={{fontFamily:"var(--font-display)"}}>Carbon projection</h4>
          <Row label="XGBoost RMSE"  value={data.projection.xgb_rmse.toFixed(2)}/>
          <Row label="XGBoost SSS"   value={data.projection.xgb_sss.toFixed(3)}/>
          <Row label="Linear-reg RMSE" value={data.projection.linreg_rmse != null ? data.projection.linreg_rmse.toFixed(2) : "—"}/>
          <Row label="Linear-reg SSS"  value={data.projection.linreg_sss  != null ? data.projection.linreg_sss.toFixed(3)  : "—"}/>
          <Row label="ARIMA RMSE"    value={data.projection.arima.rmse != null ? data.projection.arima.rmse.toFixed(2) : "—"}/>
          <Row label="ARIMA SSS"     value={data.projection.arima.sss  != null ? data.projection.arima.sss.toFixed(3)  : "—"}/>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <div className="leaf-card rounded-2xl p-6">
          <h4 className="text-lg mb-3" style={{fontFamily:"var(--font-display)"}}>Recommendation (MC-dropout)</h4>
          <Row label="MLP accuracy"   value={(data.recommendation.mlp_acc*100).toFixed(1)+" %"}/>
          <Row label="MLP F1 (macro)" value={data.recommendation.mlp_f1.toFixed(3)}/>
          <Row label="Rule-based acc" value={(data.recommendation.rule_based_acc*100).toFixed(1)+" %"}/>
          <Row label="Rule-based F1"  value={data.recommendation.rule_based_f1.toFixed(3)}/>
        </div>
      </div>

      <p className="text-xs text-[var(--stone)] mt-4">
        Metrics: <strong>RMSE / MAE</strong> (numerical accuracy) · <strong>R²</strong> (variance explained) · <strong>Spearman ρ²</strong> (rank correlation, noise-robust) · <strong>IoU</strong> (segmentation only).
        CSI removed — it mixed magnitude with correlation and was not bounded.
      </p>
    </div>
  );
}

const Row = ({label, value}) => (
  <div className="flex items-baseline justify-between py-2 border-b border-[#e6efdc] last:border-0">
    <span className="text-sm text-[var(--stone)]">{label}</span>
    <span className="mono-num text-[var(--leaf-900)]">{value}</span>
  </div>
);

const Center = ({ msg }) => (
  <div className="max-w-4xl mx-auto px-6 py-24 text-center text-[var(--stone)]">{msg}</div>
);
