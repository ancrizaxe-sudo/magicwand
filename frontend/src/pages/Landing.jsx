import { Link } from "react-router-dom";
import { ArrowRight, Trees, Brain, LineChart, Link as LinkIcon } from "lucide-react";

const claims = [
  { icon: Trees,     title: "AVLE+ index",         text: "Spatial cluster · temporal momentum · canopy density weighting, replacing raw NDVI differencing." },
  { icon: Brain,     title: "Neural recommender",  text: "4-class MLP classifier trained on 30 k synthetic samples — outputs calibrated class probabilities." },
  { icon: LineChart, title: "XGBoost projection",  text: "Scenario-conditioned 5-year forecast with BAU / mitigation divergence and 95 % quantile CI." },
  { icon: LinkIcon,  title: "On-chain ledger",     text: "Every analysis committed to a local py-evm chain — tamper-evident research provenance." },
];

export default function Landing() {
  return (
    <div className="max-w-7xl mx-auto px-6">
      <section className="pt-16 pb-24 reveal">
        <div className="grid lg:grid-cols-12 gap-10 items-start">
          <div className="lg:col-span-7">
            <p className="text-xs uppercase tracking-[0.22em] text-[var(--leaf-700)] mb-5">
              open · reproducible · platform-independent
            </p>
            <h1 data-testid="hero-title" className="text-5xl sm:text-6xl lg:text-7xl leading-[1.02]" style={{fontFamily:"var(--font-display)"}}>
              Vegetation loss,<br/>
              <span className="text-[var(--leaf-700)] italic">measured honestly.</span>
            </h1>
            <p className="mt-7 text-lg text-[#3b4a35] max-w-xl">
              AVLE-C is a research-grade remote-sensing pipeline that combines a U-Net vegetation
              segmenter, a learned carbon-flux regressor, a neural remediation recommender, and
              an XGBoost scenario forecaster — trained from scratch, running entirely on your
              hardware.
            </p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Link to="/analyze" data-testid="hero-start-btn" className="pill-btn">
                Run an analysis <ArrowRight size={16} />
              </Link>
              <Link to="/ablation" data-testid="hero-ablation-btn" className="pill-ghost">
                View ablation table
              </Link>
            </div>
          </div>

          <div className="lg:col-span-5 reveal reveal-d2">
            <div className="leaf-card rounded-3xl p-7 relative overflow-hidden">
              <span className="absolute -top-10 -right-10 h-40 w-40 rounded-full bg-[var(--leaf-100)]"></span>
              <div className="relative">
                <div className="text-xs uppercase tracking-widest text-[var(--leaf-700)]">Claim</div>
                <h3 className="mt-1 text-2xl" style={{fontFamily:"var(--font-display)"}}>Four interconnected novelties</h3>
                <div className="mt-5 divide-y divide-[#e0ecd5]">
                  {["AVLE+ weighted loss index", "ML-regressed carbon estimation",
                    "Neural recommendation classifier", "XGBoost scenario forecasting"].map((c, i) => (
                    <div key={c} className="py-3 flex items-baseline gap-3">
                      <span className="w-7 text-sm font-bold text-[var(--leaf-700)] mono-num">0{i+1}</span>
                      <span className="text-[var(--leaf-900)]">{c}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="pb-24">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
          {claims.map(({icon: Icon, title, text}, i) => (
            <div key={title} data-testid={`claim-card-${i}`}
                 className={`leaf-card rounded-2xl p-6 reveal reveal-d${i+1}`}>
              <Icon className="text-[var(--leaf-700)]" size={22} />
              <h4 className="mt-4 text-lg" style={{fontFamily:"var(--font-display)"}}>{title}</h4>
              <p className="mt-2 text-sm text-[#4b5a43] leading-relaxed">{text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="pb-24 reveal">
        <div className="leaf-card rounded-3xl p-8 lg:p-12 grid lg:grid-cols-2 gap-8 items-center">
          <div>
            <div className="text-xs uppercase tracking-[0.22em] text-[var(--leaf-700)] mb-3">pipeline</div>
            <h2 className="text-4xl" style={{fontFamily:"var(--font-display)"}}>
              From pixels to policy, in one request.
            </h2>
            <p className="mt-4 text-[#3b4a35]">
              Submit a bounding box and two date ranges. The backend fetches Sentinel-2 tiles,
              runs segmentation, computes AVLE+, estimates carbon, classifies a remediation
              action, forecasts five years ahead, and logs everything to a local EVM chain —
              synchronously, in a single call.
            </p>
          </div>
          <ol className="grid sm:grid-cols-2 gap-3 text-sm">
            {["Sentinel-2 L2A fetch", "U-Net segmentation",
              "NDVI + change detection", "AVLE+ scoring",
              "Random Forest carbon", "MLP recommendation",
              "XGBoost projection", "Blockchain receipt"].map((s, i) => (
              <li key={s} className="flex items-center gap-3 bg-[var(--leaf-50)] border border-[#d7e7ca] rounded-xl px-4 py-3">
                <span className="h-7 w-7 grid place-items-center rounded-full bg-[var(--leaf-700)] text-white text-xs mono-num">{i+1}</span>
                <span className="text-[var(--leaf-900)]">{s}</span>
              </li>
            ))}
          </ol>
        </div>
      </section>
    </div>
  );
}
