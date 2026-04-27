import { useEffect, useState } from "react";
import { getBlockchain } from "../lib/api";
import { Loader2, ShieldCheck } from "lucide-react";

export default function Ledger() {
  const [d, setD] = useState(null);
  const refresh = () => getBlockchain(50).then(setD).catch(() => setD({ status: {}, records: [] }));
  useEffect(() => { refresh(); const t = setInterval(refresh, 8000); return () => clearInterval(t); }, []);

  if (!d) return <div className="max-w-4xl mx-auto px-6 py-24 text-center text-[var(--stone)]"><Loader2 className="animate-spin inline mr-2"/>Loading …</div>;
  const s = d.status || {};

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <div className="mb-8 reveal">
        <div className="text-xs uppercase tracking-[0.22em] text-[var(--leaf-700)] mb-2">provenance</div>
        <h1 className="text-4xl sm:text-5xl" style={{fontFamily:"var(--font-display)"}}>Local EVM ledger</h1>
        <p className="mt-3 text-[#3b4a35] max-w-2xl">
          Every analysis is committed to an in-process py-evm chain. No wallet, no external node,
          no API keys — the chain runs entirely inside this backend process.
        </p>
      </div>

      <div className="grid md:grid-cols-4 gap-4 mb-8">
        <Stat label="chain id"       v={s.chain_id ?? "—"}/>
        <Stat label="latest block"   v={s.latest_block ?? "—"}/>
        <Stat label="records logged" v={s.record_count ?? 0}/>
        <Stat label="provider"       v={s.provider || "eth-tester"} small/>
      </div>

      <div className="leaf-card rounded-2xl overflow-hidden reveal reveal-d1">
        <table className="w-full text-sm" data-testid="ledger-table">
          <thead className="bg-[var(--leaf-50)] text-[var(--leaf-800)]">
            <tr>
              <th className="text-left p-4">Block</th>
              <th className="text-left p-4">Tx hash</th>
              <th className="text-left p-4">Region</th>
              <th className="text-right p-4">Carbon (tCO₂)</th>
              <th className="text-right p-4">AVLE+</th>
            </tr>
          </thead>
          <tbody>
            {d.records.length === 0 ? (
              <tr><td colSpan="5" className="p-8 text-center text-[var(--stone)]">No records yet — run an analysis to populate the ledger.</td></tr>
            ) : d.records.map((rec, i) => (
              <tr key={i} className="border-t border-[#e6efdc]">
                <td className="p-4 mono-num">{rec.proof.block_number}</td>
                <td className="p-4 font-mono text-[11px] break-all max-w-[240px]">{rec.proof.tx_hash}</td>
                <td className="p-4 font-mono text-[11px] break-all max-w-[220px]">{rec.record.region_hash}</td>
                <td className="p-4 text-right mono-num">{rec.record.carbon_estimate_tco2?.toFixed(1)}</td>
                <td className="p-4 text-right mono-num">{rec.record.avle_plus_per_km2?.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
const Stat = ({label, v, small}) => (
  <div className="leaf-card rounded-2xl p-5">
    <ShieldCheck className="text-[var(--leaf-700)]" size={18}/>
    <div className="text-xs uppercase tracking-wider text-[var(--stone)] mt-3">{label}</div>
    <div className={`${small ? "text-sm" : "text-2xl"} mt-1 mono-num break-all`} style={{fontFamily:"var(--font-display)"}}>{v}</div>
  </div>
);
