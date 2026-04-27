import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "sonner";

import NavBar from "@/components/NavBar";
import Landing from "@/pages/Landing";
import Analyze from "@/pages/Analyze";
import Ablation from "@/pages/Ablation";
import Ledger from "@/pages/Ledger";
import Datasets from "@/pages/Datasets";
import Docs from "@/pages/Docs";
import About from "@/pages/About";

export default function App() {
  return (
    <div className="App min-h-screen">
      <BrowserRouter>
        <NavBar />
        <main data-testid="app-main">
          <Routes>
            <Route path="/"          element={<Landing />} />
            <Route path="/analyze"   element={<Analyze />} />
            <Route path="/ablation"  element={<Ablation />} />
            <Route path="/ledger"    element={<Ledger />} />
            <Route path="/datasets"  element={<Datasets />} />
            <Route path="/docs"          element={<Docs />} />
            <Route path="/docs/:slug"    element={<Docs />} />
            <Route path="/about"     element={<About />} />
          </Routes>
        </main>
        <footer className="py-10 text-center text-xs text-[var(--stone)]">
          AVLE-C · open-source, platform-independent, trained from scratch.
        </footer>
        <Toaster richColors position="top-right" />
      </BrowserRouter>
    </div>
  );
}
