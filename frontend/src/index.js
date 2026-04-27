import React from "react";
import ReactDOM from "react-dom/client";
import "@/index.css";
import App from "@/App";

// StrictMode double-mounts components in development, which conflicts with
// react-leaflet 4's map-container initialization ("Map container is already
// initialized").  We disable StrictMode to avoid the false-positive crash.
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
