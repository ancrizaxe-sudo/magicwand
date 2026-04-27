import { Link, NavLink } from "react-router-dom";
import { Leaf } from "lucide-react";

const items = [
  { to: "/",          label: "Overview" },
  { to: "/analyze",   label: "Analyze" },
  { to: "/ablation",  label: "Ablation" },
  { to: "/ledger",    label: "Ledger" },
  { to: "/docs",      label: "Docs" },
  { to: "/about",     label: "About" },
];

export default function NavBar() {
  return (
    <header
      data-testid="nav-bar"
      className="sticky top-0 z-40 backdrop-blur-md bg-white/75 border-b border-[#d7e7ca]"
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" data-testid="nav-logo" className="flex items-center gap-2 group">
          <span className="h-9 w-9 grid place-items-center rounded-xl bg-[var(--leaf-700)] text-white shadow-sm">
            <Leaf size={18} strokeWidth={2.4} />
          </span>
          <div className="leading-tight">
            <div className="font-semibold tracking-tight" style={{fontFamily:"var(--font-display)"}}>AVLE-C</div>
            <div className="text-[11px] text-[var(--stone)]">Adaptive Vegetation Loss Estimator</div>
          </div>
        </Link>
        <nav className="flex items-center gap-1">
          {items.map(({to, label}) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              data-testid={`nav-link-${label.toLowerCase()}`}
              className={({isActive}) =>
                `px-3.5 py-1.5 rounded-full text-sm transition-all ${
                  isActive
                    ? "bg-[var(--leaf-700)] text-white"
                    : "text-[var(--leaf-900)] hover:bg-[var(--leaf-50)]"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
