import { useEffect, useState } from "react";
import { Download, Database, Loader2 } from "lucide-react";
import { api, API } from "../lib/api";

export default function Datasets() {
  const [d, setD] = useState(null);
  useEffect(() => { api.get("/datasets").then((r)=>setD(r.data)).catch(()=>setD({})); }, []);

  if (!d) return <div className="max-w-4xl mx-auto px-6 py-24 text-center text-[var(--stone)]"><Loader2 className="animate-spin inline mr-2"/>Loading …</div>;
  const entries = Object.entries(d.files || {});

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <div className="reveal">
        <div className="text-xs uppercase tracking-[0.22em] text-[var(--leaf-700)] mb-2">training data</div>
        <h1 className="text-4xl sm:text-5xl" style={{fontFamily:"var(--font-display)"}}>Downloadable datasets</h1>
        <p className="mt-3 text-[#3b4a35] max-w-2xl">
          All training data is fully synthetic, grounded in IPCC Tier-2 literature ranges and
          domain-parameterised rule-matrices. No proprietary inputs. Download the exact CSVs used
          to fit the four AVLE-C models below.
        </p>
      </div>

      <div className="space-y-4 mt-10">
        {entries.map(([key, meta]) => {
          const url = `${API}${meta.download_url}`;
          return (
            <div key={key} data-testid={`dataset-${key}`}
                 className="leaf-card rounded-2xl p-6 flex items-start justify-between gap-6 reveal reveal-d1">
              <div className="flex-1">
                <div className="flex items-center gap-2 text-[var(--leaf-700)]">
                  <Database size={16}/>
                  <div className="text-xs uppercase tracking-wider">{key}</div>
                </div>
                <h4 className="mt-1 text-xl font-semibold" style={{fontFamily:"var(--font-display)"}}>
                  {meta.filename}
                </h4>
                <p className="mt-2 text-sm text-[#3b4a35] max-w-2xl">{d.notes?.[key]}</p>
                <div className="mt-3 flex gap-4 text-xs text-[var(--stone)] mono-num">
                  <span>{meta.size_bytes ? (meta.size_bytes / 1024).toFixed(1) + " KB" : "—"}</span>
                </div>
              </div>
              <a data-testid={`download-${key}`} href={url} target="_blank" rel="noreferrer"
                 className="pill-btn whitespace-nowrap">
                <Download size={16}/> Download CSV
              </a>
            </div>
          );
        })}
      </div>
    </div>
  );
}
