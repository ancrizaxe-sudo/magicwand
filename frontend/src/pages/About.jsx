import { useEffect, useState } from "react";
import { getWeightsInfo } from "../lib/api";
import { BookOpen, Cpu, TreePine } from "lucide-react";

export default function About() {
  const [w, setW] = useState(null);
  useEffect(() => { getWeightsInfo().then(setW).catch(()=>setW({})); }, []);

  const carbonR2 = w?.carbon_rf?.metrics?.rf?.r2;
  const carbonRMSE = w?.carbon_rf?.metrics?.rf?.rmse;
  const recAcc = w?.recommendation_mlp?.metrics?.mlp?.accuracy;
  const recF1  = w?.recommendation_mlp?.metrics?.mlp?.f1_macro;
  const xgbRMSE = w?.xgb_projection?.metrics?.xgb?.rmse;
  const xgbSSS  = w?.xgb_projection?.metrics?.xgb?.sss;

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <div className="reveal">
        <div className="text-xs uppercase tracking-[0.22em] text-[var(--leaf-700)] mb-2">framework</div>
        <h1 className="text-4xl sm:text-5xl" style={{fontFamily:"var(--font-display)"}}>
          About AVLE-C
        </h1>
        <p className="mt-4 text-[#3b4a35] max-w-3xl">
          AVLE-C (Adaptive Vegetation Loss Estimator for Carbon) is an end-to-end, fully-open
          remote-sensing framework. Four interconnected novelty claims replace historically weak
          components of carbon estimation pipelines: raw NDVI differencing, constant IPCC
          emission factors, rule-based triage, and linear ARIMA projections.
        </p>
      </div>

      <section className="grid md:grid-cols-3 gap-4 mt-10">
        <Card icon={Cpu} title="Models trained"
              items={[
                ["U-Net (ResNet-34)", w?.segmentation_unet?.present ? "ready" : "pending"],
                ["RF carbon regressor", w?.carbon_rf?.present ? "ready" : "pending"],
                ["MLP recommender",   w?.recommendation_mlp?.present ? "ready" : "pending"],
                ["XGBoost BAU/Mit/CI",w?.xgb_projection?.bau_present ? "ready" : "pending"],
              ]}/>
        <Card icon={TreePine} title="Headline metrics"
              items={[
                ["Carbon RMSE", carbonRMSE?.toFixed(2) ?? "—"],
                ["Carbon R²",   carbonR2?.toFixed(3) ?? "—"],
                ["MLP accuracy",recAcc ? (recAcc*100).toFixed(1)+" %" : "—"],
                ["MLP F1",      recF1?.toFixed(3) ?? "—"],
                ["XGB SSS",     xgbSSS?.toFixed(3) ?? "—"],
                ["XGB RMSE",    xgbRMSE?.toFixed(2) ?? "—"],
              ]}/>
        <Card icon={BookOpen} title="Stack"
              items={[
                ["segmentation", "PyTorch"],
                ["carbon",       "scikit-learn"],
                ["recommender",  "PyTorch MLP"],
                ["forecast",     "XGBoost"],
                ["chain",        "eth-tester / py-evm"],
                ["data",         "Sentinel-2 L2A (MPC)"],
              ]}/>
      </section>

      <section className="leaf-card rounded-2xl p-8 mt-10 reveal reveal-d1">
        <h3 className="text-2xl mb-3" style={{fontFamily:"var(--font-display)"}}>
          What is <em className="italic">not</em> claimed as novel
        </h3>
        <p className="text-[#3b4a35]">
          U-Net architecture, NDVI computation, XGBoost as a model family, STAC satellite retrieval,
          FastAPI/React stack, and blockchain logging. These are engineering components. The four
          novelty claims are: <strong>AVLE+ index</strong>, <strong>ML-regressed carbon estimation</strong>,
          <strong> trained recommendation MLP</strong>, and <strong>XGBoost scenario framework</strong>.
        </p>
      </section>
    </div>
  );
}

function Card({ icon: Icon, title, items }) {
  return (
    <div className="leaf-card rounded-2xl p-6">
      <Icon className="text-[var(--leaf-700)]" size={18}/>
      <h4 className="text-lg mt-3 mb-4" style={{fontFamily:"var(--font-display)"}}>{title}</h4>
      <dl className="text-sm space-y-1.5">
        {items.map(([k,v]) => (
          <div key={k} className="flex justify-between gap-3">
            <dt className="text-[var(--stone)]">{k}</dt>
            <dd className="mono-num text-[var(--leaf-900)] text-right">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
