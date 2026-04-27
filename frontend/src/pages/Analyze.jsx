import { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, Rectangle, Marker, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Play, Satellite, Loader2, Crosshair, MousePointerClick, MapPin, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import { runAnalyze } from "../lib/api";
import ResultsPanel from "../components/ResultsPanel";

const cornerIcon = new L.DivIcon({
  className: "",
  html: '<div style="width:14px;height:14px;border-radius:50%;background:#2f6222;border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,0.3)"></div>',
  iconSize: [14, 14], iconAnchor: [7, 7],
});

const PRESETS = [
  { label: "Amazon · Pará (tropical moist)",      bbox: [-54.9, -9.3, -54.7, -9.1], t1: ["2019-06-01", "2019-08-31"], t2: ["2023-06-01", "2023-08-31"] },
  { label: "Mato Grosso · Cerrado",              bbox: [-55.2, -9.8, -54.8, -9.4], t1: ["2019-06-01", "2019-08-31"], t2: ["2023-06-01", "2023-08-31"] },
  { label: "Western Ghats · India",              bbox: [75.4, 14.2, 75.6, 14.4],   t1: ["2019-11-01", "2020-02-28"], t2: ["2023-11-01", "2024-02-28"] },
  { label: "Sumatra · Riau peatland (Indonesia)", bbox: [101.2, 0.3, 101.4, 0.5],   t1: ["2019-07-01", "2019-09-30"], t2: ["2023-07-01", "2023-09-30"] },
  { label: "Congo Basin · CAR",                  bbox: [18.4, 4.3, 18.6, 4.5],      t1: ["2019-12-01", "2020-02-28"], t2: ["2023-12-01", "2024-02-28"] },
];

const DEFAULT_BBOX = PRESETS[0].bbox;

function Recenter({ center }) {
  const map = useMap();
  useEffect(() => { if (center) map.flyTo(center, 10, { duration: 0.8 }); }, [center]);
  return null;
}
function DrawPicker({ setBbox, mode, setMode }) {
  const [first, setFirst] = useState(null);
  useMapEvents({
    click(e) {
      if (mode !== "click") return;
      const { lat, lng } = e.latlng;
      if (!first) { setFirst([lat, lng]); return; }
      const west = Math.min(first[1], lng), east = Math.max(first[1], lng);
      const south = Math.min(first[0], lat), north = Math.max(first[0], lat);
      setBbox([west, south, east, north]); setFirst(null); setMode("drag");
    },
  });
  return null;
}
function DraggableBBox({ bbox, setBbox }) {
  const [w, s, e, n] = bbox;
  const corners = [
    { id: "sw", pos: [s, w], move: ([lat, lng]) => setBbox([lng, lat, e, n]) },
    { id: "se", pos: [s, e], move: ([lat, lng]) => setBbox([w, lat, lng, n]) },
    { id: "ne", pos: [n, e], move: ([lat, lng]) => setBbox([w, s, lng, lat]) },
    { id: "nw", pos: [n, w], move: ([lat, lng]) => setBbox([lng, s, e, lat]) },
  ];
  return (
    <>
      <Rectangle bounds={[[s, w], [n, e]]} pathOptions={{ color: "#2f6222", weight: 2, fillOpacity: 0.15 }}/>
      {corners.map((c) => (
        <Marker key={c.id} position={c.pos} draggable icon={cornerIcon}
                eventHandlers={{ dragend: (ev) => { const { lat, lng } = ev.target.getLatLng(); c.move([lat, lng]); } }}/>
      ))}
    </>
  );
}

export default function Analyze() {
  const [bbox, setBbox] = useState(DEFAULT_BBOX);
  const [t1s, setT1s] = useState(PRESETS[0].t1[0]);
  const [t1e, setT1e] = useState(PRESETS[0].t1[1]);
  const [t2s, setT2s] = useState(PRESETS[0].t2[0]);
  const [t2e, setT2e] = useState(PRESETS[0].t2[1]);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [recenter, setRecenter] = useState(null);
  const [mode, setMode] = useState("drag");
  const [useSynthetic, setUseSynthetic] = useState(false);
  const resultsRef = useRef(null);

  function applyPreset(p) {
    setBbox(p.bbox);
    setT1s(p.t1[0]); setT1e(p.t1[1]);
    setT2s(p.t2[0]); setT2e(p.t2[1]);
    setRecenter([(p.bbox[1]+p.bbox[3])/2, (p.bbox[0]+p.bbox[2])/2]);
  }

  async function submit() {
    setRunning(true); setResult(null);
    try {
      const r = await runAnalyze({
        bbox, date_t1_start: t1s, date_t1_end: t1e,
        date_t2_start: t2s, date_t2_end: t2e,
        size: 384, allow_blockchain: true, use_synthetic: useSynthetic,
      });
      setResult(r);
      if (r.suitability && !r.suitability.suitable) {
        toast.warning(`Region unsuitable: ${r.suitability.reason}`);
      } else {
        toast.success(`Analysis complete (${r.source?.t2 || "source"})`);
      }
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth" }), 120);
    } catch (e) { toast.error(e?.response?.data?.detail || "Analysis failed"); }
    finally { setRunning(false); }
  }

  function locate() {
    if (!navigator.geolocation) return toast.error("Geolocation unavailable");
    toast.loading("Locating…", { id: "loc" });
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude: lat, longitude: lng } = pos.coords;
        const d = 0.2;
        setBbox([lng-d, lat-d, lng+d, lat+d]); setRecenter([lat, lng]);
        toast.success(`Centered on you (${lat.toFixed(3)}, ${lng.toFixed(3)})`, { id: "loc" });
      },
      (err) => toast.error(`Geolocation: ${err.message}`, { id: "loc" }),
      { timeout: 10000 }
    );
  }
  const center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2];

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      <div className="mb-8 reveal">
        <div className="text-xs uppercase tracking-[0.22em] text-[var(--leaf-700)] mb-2">analyze</div>
        <h1 className="text-4xl sm:text-5xl" style={{fontFamily:"var(--font-display)"}}>
          Pick a region. Pick two windows.
        </h1>
        <p className="mt-3 text-[#3b4a35] max-w-2xl">
          Drag the green corners to resize the bounding box, click twice to redraw, or choose a curated preset below.
          Vegetation is detected with the trained U-Net; NDVI is z-scored across windows to remove phenology.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 mb-5 reveal reveal-d1">
        {PRESETS.map((p, i) => (
          <button key={p.label} data-testid={`preset-${i}`}
                  onClick={() => applyPreset(p)}
                  className="pill-ghost text-xs">
            {p.label}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-7 leaf-card rounded-2xl p-4 reveal">
          <div className="flex flex-wrap items-center gap-2 mb-3">
            <button data-testid="locate-btn" onClick={locate} className="pill-ghost text-xs">
              <Crosshair size={14}/> Use my location
            </button>
            <button data-testid="mode-click-btn" onClick={() => setMode(mode === "click" ? "drag" : "click")}
                    className={mode === "click" ? "pill-btn text-xs" : "pill-ghost text-xs"}>
              <MousePointerClick size={14}/> {mode === "click" ? "Click 2 points…" : "Redraw with clicks"}
            </button>
            <div className="ml-auto text-xs text-[var(--stone)]">
              <MapPin size={12} className="inline mr-1"/> {center[0].toFixed(3)}, {center[1].toFixed(3)}
            </div>
          </div>
          <MapContainer center={center} zoom={9} scrollWheelZoom>
            <TileLayer attribution='&copy; OpenStreetMap contributors'
                       url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"/>
            <Recenter center={recenter}/>
            <DrawPicker setBbox={setBbox} mode={mode} setMode={setMode}/>
            <DraggableBBox bbox={bbox} setBbox={setBbox}/>
          </MapContainer>
          <div className="mt-3 grid grid-cols-4 gap-2 text-xs">
            {["west", "south", "east", "north"].map((lbl, i) => (
              <label key={lbl} className="flex flex-col gap-1">
                <span className="text-[var(--stone)] uppercase tracking-wider text-[10px]">{lbl}</span>
                <input data-testid={`bbox-${lbl}`} type="number" step="0.01" value={bbox[i]}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setBbox(bbox.map((b, j) => j === i ? (isNaN(v) ? b : v) : b));
                  }}
                  className="mono-num bg-[var(--leaf-50)] border border-[#d7e7ca] rounded-lg px-3 py-2 outline-none focus:border-[var(--leaf-500)]"/>
              </label>
            ))}
          </div>
        </div>

        <div className="lg:col-span-5 leaf-card rounded-2xl p-6 reveal reveal-d1">
          <h3 className="text-xl mb-5" style={{fontFamily:"var(--font-display)"}}>Temporal windows</h3>
          <div className="space-y-4">
            <DateRange label="Baseline (t₁)"    a={t1s} b={t1e} setA={setT1s} setB={setT1e} testid="t1" />
            <DateRange label="Observation (t₂)" a={t2s} b={t2e} setA={setT2s} setB={setT2e} testid="t2" />
          </div>
          <div className="mt-6 p-4 bg-[var(--leaf-50)] border border-[#d7e7ca] rounded-xl text-xs text-[var(--leaf-900)]">
            <div className="flex items-center gap-2 mb-1 font-semibold">
              <Satellite size={14} /> Data source
            </div>
            <p className="text-[#4b5a43]">
              Microsoft Planetary Computer Sentinel-2 L2A (B04 + B08). Segmentation: U-Net (ResNet-34) fine-tuned on
              matched ESA WorldCover 2021 patches. NDVI z-score normalised across t₁↔t₂.
            </p>
            <label className="mt-3 flex items-center gap-2 text-[11px] cursor-pointer select-none">
              <input data-testid="synthetic-toggle" type="checkbox" checked={useSynthetic}
                     onChange={(e) => setUseSynthetic(e.target.checked)}/>
              <span>Use deterministic synthetic scene (fast demo)</span>
            </label>
          </div>
          <button data-testid="run-analysis-btn" onClick={submit} disabled={running}
                  className="pill-btn w-full justify-center mt-6">
            {running ? <><Loader2 className="animate-spin" size={16}/> Running pipeline…</>
                     : <><Play size={16}/> Run AVLE-C analysis</>}
          </button>
        </div>
      </div>

      <div ref={resultsRef}>
        {result && result.suitability && !result.suitability.suitable && (
          <div data-testid="suitability-warning"
               className="mt-8 rounded-2xl p-5 bg-amber-50 border border-amber-300 text-amber-900 flex items-start gap-3 reveal">
            <AlertTriangle size={18} className="mt-0.5"/>
            <div>
              <div className="font-semibold">Region unsuitable for vegetation-loss analysis</div>
              <div className="text-sm">{result.suitability.reason}. Mean NDVI = {result.suitability.mean_ndvi?.toFixed(2)}.
                Try a forested bbox or pick a preset above.</div>
            </div>
          </div>
        )}
        {result && <ResultsPanel result={result} />}
      </div>
    </div>
  );
}

function DateRange({ label, a, b, setA, setB, testid }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-[var(--stone)] mb-2">{label}</div>
      <div className="grid grid-cols-2 gap-2">
        <input data-testid={`${testid}-start`} type="date" value={a} onChange={(e)=>setA(e.target.value)}
               className="mono-num bg-[var(--leaf-50)] border border-[#d7e7ca] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--leaf-500)]"/>
        <input data-testid={`${testid}-end`}   type="date" value={b} onChange={(e)=>setB(e.target.value)}
               className="mono-num bg-[var(--leaf-50)] border border-[#d7e7ca] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--leaf-500)]"/>
      </div>
    </div>
  );
}
