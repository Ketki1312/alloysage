import { useState, useEffect, useCallback } from "react";

// ─── MOCK API (replaces real fetch when backend not available) ────────────────
const MOCK_MODE = false; // set to false when running real Flask backend
const API_BASE = "http://localhost:5000/api";

const mockPredict = (composition, setName, properties) => {
  const clusters = ["Cluster_0", "Cluster_1", "Cluster_2", "Cluster_3", "Cluster_4"];
  const tempers = { "Cluster_0": "T6", "Cluster_1": "H1", "Cluster_2": "T4", "Cluster_3": "T5", "Cluster_4": "O" };
  const cluster = clusters[Math.floor(Math.random() * clusters.length)];
  return {
    cluster,
    recommended_temper: tempers[cluster],
    temper_distribution: { [tempers[cluster]]: 45, T6: 30, T4: 15, H1: 10 },
    algorithm: "GradientBoosting",
    accuracy: 97.1,
    probabilities: clusters.map((c, i) => ({
      cluster: c, temper: tempers[c],
      probability: i === clusters.indexOf(cluster) ? 0.72 : Math.random() * 0.15
    })).sort((a, b) => b.probability - a.probability)
  };
};

const mockModelInfo = {
  "Set 1 (YS & EC)": { algorithm: "GradientBoosting", accuracy: 97.1, features: ["YS (MPa)", "EC Volume (% IACS)"] },
  "Set 2 (TC & TE)": { algorithm: "RandomForest", accuracy: 85.8, features: ["TC (W/m-K)", "TE Coeff"] },
  "Set 3 (YS & Fatigue)": { algorithm: "RandomForest", accuracy: 94.2, features: ["YS (MPa)", "Fatigue Strength (MPa)"] },
};

async function apiFetch(endpoint, options = {}) {
  if (MOCK_MODE) {
    await new Promise(r => setTimeout(r, 800 + Math.random() * 600));
    if (endpoint === "/model-info") return mockModelInfo;
    if (endpoint === "/predict") {
      const b = options.body ? JSON.parse(options.body) : {};
      return mockPredict(b.composition, b.set, b.properties);
    }
    if (endpoint === "/health") return { status: "ok", models_loaded: Object.keys(mockModelInfo) };
    return {};
  }
  const res = await fetch(API_BASE + endpoint, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const ELEMENTS = ["Al", "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Ni", "Zn", "Ga", "V", "Ti"];

const PAIR_SETS = [
  { name: "Set 1 (YS & EC)", label: "Yield Strength + Electrical Conductivity", props: ["YS (MPa)", "EC Volume (% IACS)"], units: ["MPa", "% IACS"], color: "#00d4ff" },
  { name: "Set 2 (TC & TE)", label: "Thermal Conductivity + Thermal Expansion", props: ["TC (W/m-K)", "TE Coeff"], units: ["W/m·K", "×10⁻⁶/K"], color: "#ff6b35" },
  { name: "Set 3 (YS & Fatigue)", label: "Yield Strength + Fatigue Strength", props: ["YS (MPa)", "Fatigue Strength (MPa)"], units: ["MPa", "MPa"], color: "#a855f7" },
];

const TEMPER_COLORS = {
  T6: "#00d4ff", T4: "#a855f7", T5: "#ff6b35", T8: "#22c55e", H1: "#f59e0b",
  H2: "#ef4444", T62: "#06b6d4", T7: "#8b5cf6", O: "#64748b", F: "#94a3b8"
};

const DEFAULT_COMPOSITION = {
  Al: 97.9, Si: 0.6, Fe: 0.2, Cu: 0.1, Mn: 0.1, Mg: 0.8,
  Cr: 0.1, Ni: 0.0, Zn: 0.1, Ga: 0.0, V: 0.0, Ti: 0.1
};

const DEFAULT_PROPERTIES = {
  "YS (MPa)": 275, "EC Volume (% IACS)": 45, "TC (W/m-K)": 180,
  "TE Coeff": 23.0, "Fatigue Strength (MPa)": 120
};

// ─── SUB-COMPONENTS ───────────────────────────────────────────────────────────

function GlowDot({ color = "#00d4ff", size = 8 }) {
  return (
    <span style={{
      display: "inline-block", width: size, height: size, borderRadius: "50%",
      background: color, boxShadow: `0 0 ${size * 1.5}px ${color}`,
      flexShrink: 0
    }} />
  );
}

function ElementInput({ name, value, onChange }) {
  const isHighlight = ["Al", "Mg", "Zn", "Cu"].includes(name);
  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 4,
      background: isHighlight ? "rgba(0,212,255,0.04)" : "rgba(255,255,255,0.02)",
      border: `1px solid ${isHighlight ? "rgba(0,212,255,0.2)" : "rgba(255,255,255,0.07)"}`,
      borderRadius: 8, padding: "10px 12px",
    }}>
      <label style={{ fontSize: 11, fontWeight: 700, color: isHighlight ? "#00d4ff" : "#64748b", letterSpacing: 2, textTransform: "uppercase" }}>
        {name}
      </label>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <input
          type="number" step="0.01" min="0" max="100"
          value={value}
          onChange={e => onChange(name, e.target.value)}
          style={{
            background: "transparent", border: "none", outline: "none",
            color: "#f1f5f9", fontSize: 16, fontWeight: 600, width: "100%",
            fontFamily: "'JetBrains Mono', monospace",
          }}
        />
        <span style={{ fontSize: 10, color: "#475569" }}>wt%</span>
      </div>
    </div>
  );
}

function PropertyInput({ name, value, unit, onChange, accentColor }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 6,
      background: "rgba(255,255,255,0.03)",
      border: `1px solid ${accentColor}40`,
      borderRadius: 10, padding: "12px 16px",
    }}>
      <label style={{ fontSize: 11, fontWeight: 700, color: accentColor, letterSpacing: 1.5, textTransform: "uppercase" }}>
        {name.replace(" (MPa)", "").replace(" (% IACS)", "").replace(" (W/m-K)", "")}
      </label>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <input
          type="number" step="any" min="0"
          value={value}
          onChange={e => onChange(name, e.target.value)}
          style={{
            background: "transparent", border: "none", outline: "none",
            color: "#f1f5f9", fontSize: 20, fontWeight: 700, width: "100%",
            fontFamily: "'JetBrains Mono', monospace",
          }}
        />
        <span style={{ fontSize: 12, color: "#475569", whiteSpace: "nowrap" }}>{unit}</span>
      </div>
    </div>
  );
}

function TemperBadge({ temper, size = "md" }) {
  const color = TEMPER_COLORS[temper] || "#94a3b8";
  const pad = size === "lg" ? "12px 24px" : "5px 12px";
  const fs = size === "lg" ? 22 : 13;
  return (
    <span style={{
      background: `${color}20`, border: `1.5px solid ${color}`,
      color: color, borderRadius: 6, padding: pad, fontSize: fs,
      fontWeight: 800, fontFamily: "'JetBrains Mono', monospace",
      boxShadow: `0 0 12px ${color}30`, letterSpacing: 2,
    }}>
      {temper}
    </span>
  );
}

function ResultCard({ result, setInfo }) {
  const [expanded, setExpanded] = useState(false);
  if (!result) return null;
  const accent = setInfo?.color || "#00d4ff";

  return (
    <div style={{
      background: "rgba(0,0,0,0.4)", border: `1px solid ${accent}40`,
      borderRadius: 16, overflow: "hidden",
      boxShadow: `0 0 40px ${accent}10`, animation: "fadeSlide 0.4s ease",
    }}>
      <div style={{
        background: `linear-gradient(135deg, ${accent}15, transparent)`,
        padding: "24px 28px",
        borderBottom: `1px solid ${accent}20`,
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
          <div>
            <p style={{ color: "#64748b", fontSize: 12, letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>
              Recommended Heat Treatment
            </p>
            <TemperBadge temper={result.recommended_temper} size="lg" />
          </div>
          <div style={{ textAlign: "right" }}>
            <p style={{ color: "#64748b", fontSize: 11, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 4 }}>
              Alloy Family
            </p>
            <p style={{ color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace", fontSize: 16, fontWeight: 600 }}>
              {result.cluster}
            </p>
            <p style={{ color: "#475569", fontSize: 11, marginTop: 4 }}>
              via {result.algorithm} · {result.accuracy}% accuracy
            </p>
          </div>
        </div>
      </div>

      {result.probabilities && (
        <div style={{ padding: "20px 28px" }}>
          <p style={{ color: "#64748b", fontSize: 11, letterSpacing: 2, textTransform: "uppercase", marginBottom: 14 }}>
            Cluster Probability Distribution
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {result.probabilities.slice(0, 5).map((p, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span style={{ width: 70, fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: i === 0 ? accent : "#64748b" }}>
                  {p.cluster.replace("Cluster_", "C-")}
                </span>
                <TemperBadge temper={p.temper} />
                <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", borderRadius: 4, height: 6, overflow: "hidden" }}>
                  <div style={{
                    width: `${p.probability * 100}%`, height: "100%",
                    background: i === 0 ? accent : "#334155", borderRadius: 4,
                    transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
                  }} />
                </div>
                <span style={{ width: 48, textAlign: "right", fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: i === 0 ? accent : "#475569" }}>
                  {(p.probability * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {result.temper_distribution && Object.keys(result.temper_distribution).length > 0 && (
        <div style={{ padding: "0 28px 20px" }}>
          <button
            onClick={() => setExpanded(!expanded)}
            style={{
              background: "none", border: `1px solid ${accent}30`, borderRadius: 6,
              color: accent, fontSize: 12, padding: "6px 14px", cursor: "pointer",
              letterSpacing: 1, textTransform: "uppercase",
            }}
          >
            {expanded ? "▲ Hide" : "▼ Show"} Temper Distribution in Cluster
          </button>
          {expanded && (
            <div style={{ marginTop: 14, display: "flex", flexWrap: "wrap", gap: 10 }}>
              {Object.entries(result.temper_distribution)
                .sort((a, b) => b[1] - a[1])
                .map(([t, count]) => (
                  <div key={t} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <TemperBadge temper={t} />
                    <span style={{ color: "#64748b", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }}>{count}</span>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ModelInfoPanel({ modelInfo }) {
  if (!modelInfo) return null;
  return (
    <div style={{
      background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.07)",
      borderRadius: 12, padding: "18px 22px",
    }}>
      <p style={{ color: "#64748b", fontSize: 11, letterSpacing: 2, textTransform: "uppercase", marginBottom: 14 }}>
        Trained Models
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {PAIR_SETS.map(ps => {
          const info = modelInfo[ps.name];
          if (!info) return null;
          return (
            <div key={ps.name} style={{
              display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap"
            }}>
              <GlowDot color={ps.color} />
              <span style={{ color: "#94a3b8", fontSize: 12, flex: 1, minWidth: 120 }}>
                {ps.name}
              </span>
              <span style={{ color: "#475569", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                {info.algorithm}
              </span>
              <span style={{
                color: info.accuracy > 90 ? "#22c55e" : info.accuracy > 70 ? "#f59e0b" : "#ef4444",
                fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 700
              }}>
                {info.accuracy}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function AlloySage() {
  const [activeTab, setActiveTab] = useState("predict");
  const [selectedSet, setSelectedSet] = useState(PAIR_SETS[0].name);
  const [composition, setComposition] = useState(DEFAULT_COMPOSITION);
  const [propValues, setPropValues] = useState(DEFAULT_PROPERTIES);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelInfo, setModelInfo] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);

  const currentSet = PAIR_SETS.find(p => p.name === selectedSet);

  useEffect(() => {
    apiFetch("/model-info").then(setModelInfo).catch(() => {});
    apiFetch("/health").then(setHealthStatus).catch(() => {});
  }, []);

  const updateComposition = useCallback((name, val) => {
    setComposition(prev => ({ ...prev, [name]: parseFloat(val) || 0 }));
  }, []);

  const updateProp = useCallback((name, val) => {
    setPropValues(prev => ({ ...prev, [name]: parseFloat(val) || 0 }));
  }, []);

  const handlePredict = async () => {
    setLoading(true); setError(""); setResult(null);
    const properties = {};
    for (const prop of currentSet.props) {
      properties[prop] = propValues[prop] ?? 0;
    }
    try {
      const res = await apiFetch("/predict", {
        method: "POST",
        body: JSON.stringify({ composition, set: selectedSet, properties }),
      });
      setResult(res);
    } catch (e) {
      setError(e.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const totalWt = Object.values(composition).reduce((a, b) => a + (parseFloat(b) || 0), 0);
  const wtOk = Math.abs(totalWt - 100) < 0.5;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#060a14",
      color: "#e2e8f0",
      fontFamily: "'IBM Plex Sans', 'Helvetica Neue', sans-serif",
      position: "relative",
      overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input::-webkit-inner-spin-button, input::-webkit-outer-spin-button { -webkit-appearance: none; }
        input[type=number] { -moz-appearance: textfield; }
        @keyframes fadeSlide {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse { 0%,100% { opacity:0.6; } 50% { opacity:1; } }
        @keyframes scan {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
      `}</style>

      {/* Decorative grid bg */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none",
        backgroundImage: "linear-gradient(rgba(0,212,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px)",
        backgroundSize: "40px 40px",
        zIndex: 0,
      }} />

      {/* Scan line animation */}
      <div style={{
        position: "fixed", left: 0, right: 0, height: 1,
        background: "linear-gradient(90deg,transparent,rgba(0,212,255,0.15),transparent)",
        animation: "scan 8s linear infinite", zIndex: 0, pointerEvents: "none",
      }} />

      <div style={{ position: "relative", zIndex: 1, maxWidth: 1200, margin: "0 auto", padding: "0 24px 60px" }}>

        {/* Header */}
        <header style={{ padding: "32px 0 28px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
                <div style={{
                  width: 36, height: 36, borderRadius: 8,
                  background: "linear-gradient(135deg, #00d4ff, #0066cc)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 18, boxShadow: "0 0 24px rgba(0,212,255,0.4)"
                }}>⬡</div>
                <h1 style={{
                  fontSize: 26, fontWeight: 700, letterSpacing: -0.5,
                  background: "linear-gradient(90deg, #e2e8f0, #94a3b8)",
                  WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
                }}>
                  AlloySage
                </h1>
                <span style={{
                  background: "rgba(0,212,255,0.15)", border: "1px solid rgba(0,212,255,0.3)",
                  color: "#00d4ff", fontSize: 10, padding: "3px 8px", borderRadius: 4,
                  fontWeight: 700, letterSpacing: 2, textTransform: "uppercase"
                }}>
                  {MOCK_MODE ? "DEMO" : "LIVE"}
                </span>
              </div>
              <p style={{ color: "#475569", fontSize: 13, letterSpacing: 0.3 }}>
                Wrought Aluminium Alloy · Temper Prediction Engine · VNIT MMT Project
              </p>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <GlowDot color={healthStatus?.status === "ok" ? "#22c55e" : "#f59e0b"} size={7} />
              <span style={{ fontSize: 12, color: "#475569" }}>
                {healthStatus ? `${healthStatus.models_loaded?.length || 0} models loaded` : "Connecting…"}
              </span>
            </div>
          </div>
        </header>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginTop: 24, marginBottom: 32, borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 1 }}>
          {[["predict", "⬡ Predict"], ["models", "◈ Models"], ["about", "◎ About"]].map(([key, label]) => (
            <button key={key} onClick={() => setActiveTab(key)} style={{
              background: "none", border: "none", cursor: "pointer",
              color: activeTab === key ? "#00d4ff" : "#475569",
              fontSize: 13, fontWeight: 600, padding: "10px 20px", letterSpacing: 1,
              borderBottom: activeTab === key ? "2px solid #00d4ff" : "2px solid transparent",
              transition: "all 0.2s",
            }}>{label}</button>
          ))}
        </div>

        {/* ── TAB: PREDICT ── */}
        {activeTab === "predict" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 28, alignItems: "start" }}>

            {/* LEFT: Inputs */}
            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

              {/* Pair Set Selector */}
              <div>
                <p style={{ color: "#64748b", fontSize: 11, letterSpacing: 2, textTransform: "uppercase", marginBottom: 14 }}>
                  Step 1 · Select Property Pair
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {PAIR_SETS.map(ps => (
                    <button key={ps.name} onClick={() => { setSelectedSet(ps.name); setResult(null); }} style={{
                      background: selectedSet === ps.name ? `${ps.color}12` : "rgba(0,0,0,0.3)",
                      border: `1.5px solid ${selectedSet === ps.name ? ps.color : "rgba(255,255,255,0.07)"}`,
                      borderRadius: 10, padding: "14px 18px", cursor: "pointer",
                      display: "flex", alignItems: "center", gap: 14, textAlign: "left",
                      transition: "all 0.2s",
                    }}>
                      <GlowDot color={ps.color} size={9} />
                      <div>
                        <p style={{ color: selectedSet === ps.name ? ps.color : "#94a3b8", fontSize: 13, fontWeight: 600 }}>
                          {ps.name}
                        </p>
                        <p style={{ color: "#475569", fontSize: 11, marginTop: 2 }}>{ps.label}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Property Values */}
              <div>
                <p style={{ color: "#64748b", fontSize: 11, letterSpacing: 2, textTransform: "uppercase", marginBottom: 14 }}>
                  Step 2 · Enter Property Values
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {currentSet?.props.map((prop, i) => (
                    <PropertyInput
                      key={prop} name={prop}
                      value={propValues[prop] ?? ""}
                      unit={currentSet.units[i]}
                      onChange={updateProp}
                      accentColor={currentSet.color}
                    />
                  ))}
                </div>
              </div>

              {/* Composition */}
              <div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
                  <p style={{ color: "#64748b", fontSize: 11, letterSpacing: 2, textTransform: "uppercase" }}>
                    Step 3 · Alloy Composition (wt%)
                  </p>
                  <span style={{
                    fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 700,
                    color: wtOk ? "#22c55e" : "#ef4444",
                    background: wtOk ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
                    border: `1px solid ${wtOk ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
                    borderRadius: 6, padding: "3px 10px",
                  }}>
                    Σ = {totalWt.toFixed(2)}%
                  </span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
                  {ELEMENTS.map(el => (
                    <ElementInput key={el} name={el} value={composition[el] ?? 0} onChange={updateComposition} />
                  ))}
                </div>
              </div>

              {/* Predict Button */}
              <button
                onClick={handlePredict}
                disabled={loading || !wtOk}
                style={{
                  background: loading ? "rgba(0,212,255,0.1)" : "linear-gradient(135deg, #00d4ff, #0066ff)",
                  border: loading ? "1px solid rgba(0,212,255,0.3)" : "none",
                  borderRadius: 12, padding: "16px 32px", cursor: loading || !wtOk ? "not-allowed" : "pointer",
                  color: loading ? "#00d4ff" : "#060a14", fontSize: 15, fontWeight: 700, letterSpacing: 1.5,
                  textTransform: "uppercase", transition: "all 0.2s", opacity: !wtOk ? 0.5 : 1,
                  boxShadow: loading || !wtOk ? "none" : "0 0 30px rgba(0,212,255,0.3)",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                }}
              >
                {loading ? (
                  <>
                    <span style={{ animation: "pulse 1s infinite" }}>◈</span> Analysing…
                  </>
                ) : (
                  <>⬡ Predict Temper</>
                )}
              </button>

              {!wtOk && (
                <p style={{ color: "#ef4444", fontSize: 12, textAlign: "center" }}>
                  ⚠ Composition must sum to 100%. Current: {totalWt.toFixed(2)}%
                </p>
              )}

              {error && (
                <div style={{
                  background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)",
                  borderRadius: 10, padding: "14px 18px", color: "#fca5a5", fontSize: 13,
                }}>
                  ✕ {error}
                </div>
              )}
            </div>

            {/* RIGHT: Results */}
            <div style={{ display: "flex", flexDirection: "column", gap: 20, position: "sticky", top: 24 }}>
              <ModelInfoPanel modelInfo={modelInfo} />
              {result && <ResultCard result={result} setInfo={currentSet} />}
              {!result && !loading && (
                <div style={{
                  background: "rgba(0,0,0,0.3)", border: "1px dashed rgba(255,255,255,0.08)",
                  borderRadius: 16, padding: "48px 28px", textAlign: "center",
                  color: "#334155",
                }}>
                  <div style={{ fontSize: 40, marginBottom: 16 }}>⬡</div>
                  <p style={{ fontSize: 13 }}>Configure inputs and click Predict</p>
                  <p style={{ fontSize: 12, marginTop: 6 }}>K-Means cluster analysis + heat treatment recommendation</p>
                </div>
              )}
              {loading && (
                <div style={{
                  background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,212,255,0.15)",
                  borderRadius: 16, padding: "48px 28px", textAlign: "center",
                }}>
                  <div style={{ fontSize: 40, marginBottom: 16, animation: "pulse 1s infinite", color: "#00d4ff" }}>◈</div>
                  <p style={{ color: "#00d4ff", fontSize: 13, letterSpacing: 2 }}>CLASSIFYING ALLOY</p>
                  <p style={{ color: "#475569", fontSize: 12, marginTop: 8 }}>Running ensemble classifier…</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── TAB: MODELS ── */}
        {activeTab === "models" && (
          <div style={{ maxWidth: 680 }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 24, color: "#94a3b8" }}>Model Details</h2>
            {PAIR_SETS.map(ps => {
              const info = modelInfo?.[ps.name];
              return (
                <div key={ps.name} style={{
                  background: "rgba(0,0,0,0.4)", border: `1px solid ${ps.color}25`,
                  borderRadius: 14, padding: "24px 28px", marginBottom: 16,
                  borderLeft: `3px solid ${ps.color}`,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                    <GlowDot color={ps.color} size={10} />
                    <div>
                      <h3 style={{ fontSize: 15, fontWeight: 700, color: "#e2e8f0" }}>{ps.name}</h3>
                      <p style={{ color: "#64748b", fontSize: 12, marginTop: 2 }}>{ps.label}</p>
                    </div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                    {[
                      ["Input Features", `12 elements + ${ps.props.join(", ")}`],
                      ["Algorithm", info?.algorithm || "—"],
                      ["Test Accuracy", info ? `${info.accuracy}%` : "—"],
                    ].map(([label, val]) => (
                      <div key={label} style={{
                        background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "12px 14px"
                      }}>
                        <p style={{ color: "#475569", fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 6 }}>{label}</p>
                        <p style={{
                          color: label === "Test Accuracy" ? (info?.accuracy > 90 ? "#22c55e" : "#f59e0b") : "#94a3b8",
                          fontFamily: "'JetBrains Mono', monospace", fontSize: 13, fontWeight: 600
                        }}>{val}</p>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop: 14, padding: "12px 14px", background: "rgba(255,255,255,0.02)", borderRadius: 8 }}>
                    <p style={{ color: "#475569", fontSize: 11, lineHeight: 1.7 }}>
                      <span style={{ color: "#64748b" }}>Pipeline: </span>
                      Composition (2xxx/7xxx filtered) → K-Means(k=5, StandardScaler) → 
                      RandomizedSearchCV(RF/GB/XGB, 5 iter, 3-fold CV) → Cluster → Most-common Base Temper
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ── TAB: ABOUT ── */}
        {activeTab === "about" && (
          <div style={{ maxWidth: 640, display: "flex", flexDirection: "column", gap: 20 }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#94a3b8" }}>About AlloySage</h2>
            {[
              ["What is this?", "AlloySage predicts the recommended heat treatment temper (e.g. T6, H1, T4) for wrought aluminium alloys given their elemental composition and two material properties."],
              ["How it works", "The system uses K-Means clustering (k=5) to group alloys by property similarity, then trains gradient-boosted, random forest, and XGBoost classifiers to predict which cluster a new alloy belongs to. The recommended temper is the most common heat treatment observed in that cluster."],
              ["Three Pair Sets", "Set 1 uses YS + EC; Set 2 uses TC + TE; Set 3 uses YS + Fatigue Strength. Each set is independently trained and can yield different but related recommendations."],
              ["Data", "868 wrought alloy records. 2xxx and 7xxx series are excluded from training (high Cu/Zn alloys form distinct families that would dominate cluster assignment). 12 elemental features: Al, Si, Fe, Cu, Mn, Mg, Cr, Ni, Zn, Ga, V, Ti."],
              ["REST API", 'Flask backend at localhost:5000. Endpoints: GET /api/health, GET /api/model-info, POST /api/predict, POST /api/predict-all-sets, POST /api/load. See app.py for full schema.'],
            ].map(([title, body]) => (
              <div key={title} style={{
                background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: 12, padding: "20px 22px",
              }}>
                <h3 style={{ color: "#00d4ff", fontSize: 13, fontWeight: 700, marginBottom: 8, letterSpacing: 1 }}>{title}</h3>
                <p style={{ color: "#64748b", fontSize: 13, lineHeight: 1.8 }}>{body}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}