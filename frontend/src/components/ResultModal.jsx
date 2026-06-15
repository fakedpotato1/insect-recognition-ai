import { useRef, useEffect, useState } from "react";
import { X, AlertTriangle, XCircle, ChevronDown, ChevronUp, Zap, TrendingUp, Star, TrendingDown, StarOff } from "lucide-react";
import { useInsectData } from "../api/useInsectData";

// ─── Design tokens (mirrors InsectEncyclopedia) ───────────────────────────────
const ORDER_COLORS = {
  Hymenoptera: "#8B5CF6",
  Diptera:     "#0EA5E9",
  Coleoptera:  "#F59E0B",
  Blattodea:   "#10B981",
  Orthoptera:  "#22C55E",
  Lepidoptera: "#EC4899",
  Hemiptera:   "#EF4444",
  Araneae:     "#6B7280",
  Zygentoma:   "#06B6D4",
  Stylommatophora: "#A16207",
};

// Derive status + palette from the JSON "impact" field
function resolveInsectStyle(insectData) {
  if (!insectData) return null;

  const imp = (insectData.impact || "").toLowerCase();
  const isBeneficial = imp.startsWith("beneficial");
  const isHarmful    = imp.startsWith("harmful");
  const isBoth       = imp === "both";

  const status = isBoth ? "both" : isBeneficial ? "beneficial" : "harmful";

  const PALETTES = {
    harmful:    { statusBg: "bg-red-100",   statusText: "text-red-700",   statusBorder: "border-red-200",   dot: "bg-red-500"   },
    beneficial: { statusBg: "bg-green-100", statusText: "text-green-700", statusBorder: "border-green-200", dot: "bg-green-500" },
    both:       { statusBg: "bg-amber-100", statusText: "text-amber-700", statusBorder: "border-amber-200", dot: "bg-amber-500" },
  };

  // Map insect name → encyclopedia colour/bg/emoji
  const INSECT_THEMES = {
    bee:           { color: "#F59E0B", bg: "#FEF3C7", emoji: "🐝" },
    spider:         { color: "#6B7280", bg: "#F3F4F6", emoji: "🕷️" },
    ant:            { color: "#B45309", bg: "#FEF3C7", emoji: "🐜" },
    bedbug:         { color: "#DC2626", bg: "#FEE2E2", emoji: "🪲" },
    "bed-bug":      { color: "#DC2626", bg: "#FEE2E2", emoji: "🪲" },
    beetle:         { color: "#7C3AED", bg: "#EDE9FE", emoji: "🪲" },
    bernsteinschabe:{ color: "#D97706", bg: "#FEF3C7", emoji: "🪳" },
    cockroach:      { color: "#92400E", bg: "#FEF3C7", emoji: "🪳" },
    fly:            { color: "#65A30D", bg: "#ECFCCB", emoji: "🪰" },
    fruitfly:       { color: "#EA580C", bg: "#FFEDD5", emoji: "🪰" },
    "fruit fly":    { color: "#EA580C", bg: "#FFEDD5", emoji: "🪰" },
    grasshopper:    { color: "#16A34A", bg: "#DCFCE7", emoji: "🦗" },
    hornet:         { color: "#D97706", bg: "#FEF3C7", emoji: "🐝" },
    housefly:       { color: "#374151", bg: "#F3F4F6", emoji: "🪰" },
    ladybug:        { color: "#DC2626", bg: "#FEE2E2", emoji: "🐞" },
    mosquito:       { color: "#0369A1", bg: "#E0F2FE", emoji: "🦟" },
    moth:           { color: "#7C3AED", bg: "#EDE9FE", emoji: "🦋" },
    silverfish:     { color: "#0284C7", bg: "#E0F2FE", emoji: "🐛" },
    slug:           { color: "#4B5563", bg: "#F3F4F6", emoji: "🐌" },
    snail:          { color: "#92400E", bg: "#FEF3C7", emoji: "🐌" },
    tiger_mosquito:  { color: "#1D4ED8", bg: "#EFF6FF", emoji: "🦟" },
    "tiger_mosquito":{ color: "#1D4ED8", bg: "#EFF6FF", emoji: "🦟" },
    wasp:           { color: "#CA8A04", bg: "#FEF9C3", emoji: "🐝" },
  };

  const key = insectData.key || insectData.name?.toLowerCase().replace(/[\s\-]+/g, "");
  const theme = INSECT_THEMES[key] || INSECT_THEMES[insectData.name?.toLowerCase()] || {
    color: "#6366F1", bg: "#EEF2FF", emoji: "🐛",
  };

  return { status, palette: PALETTES[status], ...theme };
}

// ─── Confetti ─────────────────────────────────────────────────────────────────
function ConfettiBurst() {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    const pieces = Array.from({ length: 120 }, () => {
      const angle = Math.random() * 2 * Math.PI;
      const speed = Math.random() * 7 + 3;
      return {
        x: canvas.width / 2, y: canvas.height / 2,
        r: Math.random() * 6 + 3,
        color: ["#0e7490","#06b6d4","#a5f3fc","#166534","#bbf7d0","#fbbf24","#f87171","#c084fc","#34d399"][
          Math.floor(Math.random() * 9)
        ],
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 4,
        gravity: 0.18, alpha: 1,
        shape: Math.random() > 0.5 ? "rect" : "circle",
      };
    });
    let frame;
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      let alive = false;
      pieces.forEach((p) => {
        p.vy += p.gravity; p.x += p.vx; p.y += p.vy; p.alpha -= 0.012;
        if (p.alpha <= 0) return;
        alive = true;
        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = p.color;
        if (p.shape === "rect") ctx.fillRect(p.x - p.r / 2, p.y - p.r / 2, p.r, p.r * 1.6);
        else { ctx.beginPath(); ctx.arc(p.x, p.y, p.r / 2, 0, Math.PI * 2); ctx.fill(); }
      });
      ctx.globalAlpha = 1;
      if (alive) frame = requestAnimationFrame(animate);
    };
    animate();
    return () => cancelAnimationFrame(frame);
  }, []);
  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none rounded-2xl" />;
}

// ─── Confidence level tag ─────────────────────────────────────────────────────
function ConfidenceTag({ pct }) {
  if (pct >= 85) {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-200 text-green-900 border border-green-200">
        <Star size={10} /> Very High
      </span>
    );
  }
  if (pct >= 60) {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-green-100 text-green-700 border border-green-200">
        <TrendingUp size={10} /> High
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-yellow-100 text-yellow-700 border border-yellow-200">
      <TrendingDown size={10} /> Medium
    </span>
  );
}

// ─── Confidence bar ───────────────────────────────────────────────────────────
function ConfidenceBar({ confidence, accentColor }) {
  const pct = parseFloat((confidence * 100).toFixed(1));
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold uppercase tracking-widest" style={{ color: accentColor }}>
          AI Confidence
        </span>
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-black" style={{ color: accentColor }}>{pct}%</span>
          <ConfidenceTag pct={pct} />
        </div>
      </div>
      <div className="relative w-full h-2.5 rounded-full" style={{ background: "rgba(0,0,0,0.08)" }}>
        <div
          className="h-2.5 rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: accentColor }}
        />
      </div>
    </div>
  );
}

// ─── Stat pill (mirrors encyclopedia StatPill) ────────────────────────────────
function StatPill({ label, value, bg, accentColor }) {
  return (
    <div
      className="flex flex-col items-center justify-center rounded-xl p-2.5 border min-w-0"
      style={{ background: "rgba(255,255,255,0.65)", borderColor: "rgba(255,255,255,0.9)" }}
    >
      <span className="text-xs font-medium uppercase tracking-wide truncate w-full text-center text-gray-500">{label}</span>
      <span className="text-sm font-bold mt-0.5 truncate w-full text-center text-gray-800">{value}</span>
    </div>
  );
}

// ─── Expanded detail rows ─────────────────────────────────────────────────────
function DetailRow({ icon, label, value }) {
  return (
    <div className="flex gap-3 p-3 rounded-xl bg-white bg-opacity-60">
      <span className="text-lg shrink-0 mt-0.5">{icon}</span>
      <div className="min-w-0">
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">{label}</div>
        <div className="text-sm text-gray-800 mt-0.5 leading-snug">{value}</div>
      </div>
    </div>
  );
}

// ─── The insect card shown inside the modal ───────────────────────────────────
function InsectResultCard({ insectData, style, confidence, expanded, onToggle, onClose }) {
  const { color, bg, emoji, palette } = style;
  const orderColor = ORDER_COLORS[insectData.order] || "#6B7280";

  return (
    <div className="rounded-2xl overflow-hidden shadow-lg" style={{ background: bg }}>

      {/* Accent bar */}
      <div className="h-2 w-full" style={{ background: color }} />

      <div className="p-5">

        {/* Detection Result header row */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400">AI Classification</p>
            <h3
              className="text-lg font-black leading-tight tracking-tight"
              style={{ color }}
            >
              Detection Result
            </h3>
          </div>
          <button
            onClick={onClose}
            className="h-8 w-8 flex items-center justify-center rounded-full transition-colors"
            style={{ background: `${color}18`, color }}
          >
            <X size={15} />
          </button>
        </div>

        {/* Insect identity row */}
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex items-center gap-3">
            <span className="text-5xl leading-none select-none">{emoji}</span>
            <div>
              <h2 className="font-black text-gray-900 text-xl leading-tight">{insectData.name}</h2>
              <p className="text-xs text-gray-500 italic mt-0.5">{insectData.specificName}</p>
            </div>
          </div>
          {/* Status pill */}
          <span
            className={`shrink-0 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${palette.statusBg} ${palette.statusText} ${palette.statusBorder}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${palette.dot}`} />
            {insectData.impact.split("(")[0].trim()}
          </span>
        </div>

        {/* Order badge */}
        <div className="mb-3">
          <span
            className="inline-block text-xs font-semibold px-2.5 py-0.5 rounded-full text-white"
            style={{ background: orderColor }}
          >
            {insectData.order}
          </span>
        </div>

        {/* Confidence bar */}
        <div className="mb-3">
          <ConfidenceBar confidence={confidence} accentColor={color} />
        </div>

        {/* Quick stats */}
        <div className="grid grid-cols-3 gap-2 mb-3">
          <StatPill label="Size"     value={insectData.size}         />
          <StatPill label="Lifespan" value={insectData.lifespan}     />
          <StatPill label="Species"  value={insectData.totalSpecies} />
        </div>

        {/* Habitat (always visible) */}
        <p className="text-xs text-gray-600 line-clamp-1">
          <span className="font-semibold text-gray-700">📍 Habitat: </span>{insectData.habitat}
        </p>

        {/* Expanded section */}
        {expanded && (
          <div className="mt-4 space-y-2">
            <DetailRow icon="🍽️" label="Diet"           value={insectData.diet} />
            <DetailRow icon="🔍" label="How to Identify" value={insectData.identificationFeatures} />

            {/* Impact box */}
            <div
              className="mt-1 p-4 rounded-2xl border-2"
              style={{ borderColor: color, background: "rgba(255,255,255,0.5)" }}
            >
              <div className="text-xs font-bold uppercase tracking-wide mb-1" style={{ color }}>
                Impact on Humans
              </div>
              <div className="text-sm font-medium text-gray-800">{insectData.impact}</div>
            </div>
          </div>
        )}

        {/* Expand / collapse toggle */}
        <button
          onClick={onToggle}
          className="mt-4 w-full flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-bold transition-all"
          style={{
            background: "rgba(255,255,255,0.55)",
            color,
            border: `1.5px solid ${color}33`,
          }}
        >
          {expanded ? (
            <><ChevronUp size={14} /> Show less</>
          ) : (
            <><ChevronDown size={14} /> Learn more</>
          )}
        </button>
      </div>
    </div>
  );
}

// ─── Detection failed card ────────────────────────────────────────────────────
function FailedCard({ onClose }) {
  return (
    <div className="rounded-2xl overflow-hidden shadow-lg bg-red-50 border border-red-200">
      <div className="h-2 w-full bg-red-500" />
      <div className="p-5">
        {/* Header row */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400">AI Classification</p>
            <h3 className="text-lg font-black leading-tight tracking-tight text-red-600">Detection Result</h3>
          </div>
          <button
            onClick={onClose}
            className="h-8 w-8 flex items-center justify-center rounded-full bg-red-100 text-red-500 transition-colors hover:bg-red-200"
          >
            <X size={15} />
          </button>
        </div>
        <div className="text-center pb-2">
          <div className="flex justify-center mb-4">
            <div className="h-16 w-16 rounded-full bg-red-100 flex items-center justify-center">
              <XCircle size={36} className="text-red-500" />
            </div>
          </div>
          <h2 className="text-xl font-black text-gray-900 mb-2">Detection Failed</h2>
          <p className="text-sm text-gray-500 leading-relaxed">
            We couldn't confidently identify the insect. Please try again with a
            clearer, well-lit photo where the insect is fully visible.
          </p>
          <button
            onClick={onClose}
            className="mt-6 w-full py-2.5 bg-red-500 hover:bg-red-600 text-white font-bold rounded-xl transition-colors text-sm"
          >
            Try Again
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Uncertain banner ─────────────────────────────────────────────────────────
function UncertainBanner() {
  return (
    <div className="flex items-start gap-2 px-3 py-2 bg-yellow-50 border border-yellow-200 rounded-xl text-yellow-700 text-xs mb-3">
      <AlertTriangle size={13} className="shrink-0 mt-0.5" />
      <span>Low confidence result — consider retaking with a clearer image.</span>
    </div>
  );
}

// ─── Unknown insect fallback ──────────────────────────────────────────────────
function UnknownCard({ insectName, confidence, onClose }) {
  const pct = confidence * 100;
  return (
    <div className="rounded-2xl overflow-hidden shadow-lg bg-indigo-50 border border-indigo-100">
      <div className="h-2 w-full bg-indigo-500" />
      <div className="p-5">
        {/* Header row */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400">AI Classification</p>
            <h3 className="text-lg font-black leading-tight tracking-tight text-indigo-600">Detection Result</h3>
          </div>
          <button
            onClick={onClose}
            className="h-8 w-8 flex items-center justify-center rounded-full bg-indigo-100 text-indigo-500 transition-colors hover:bg-indigo-200"
          >
            <X size={15} />
          </button>
        </div>
        <div className="text-center">
          <span className="text-5xl">🐛</span>
          <h2 className="font-black text-xl text-gray-900 mt-2">{insectName}</h2>
          <p className="text-xs text-indigo-400 italic mt-0.5">Species not in knowledge base</p>
          <div className="mt-4 text-left">
            <ConfidenceBar confidence={confidence} accentColor="#6366F1" />
          </div>
          {pct < 60 && <div className="mt-3"><UncertainBanner /></div>}
          <button
            onClick={onClose}
            className="mt-5 w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl transition-colors text-sm"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Main ResultModal ─────────────────────────────────────────────────────────
function ResultModal({ result, onClose }) {
  const { lookupInsect } = useInsectData();
  const [expanded, setExpanded] = useState(false);

  if (!result) return null;

  const pct = result.confidence * 100;
  const insectData = lookupInsect(result.insect_name);
  const style = resolveInsectStyle(insectData);
  const showConfetti = pct >= 60 && !!insectData;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/50 backdrop-blur-sm px-3 pb-4 sm:px-4 sm:pb-0"
      onClick={onClose}
    >
      {/* Modal container — hidden scrollbar, rounded everywhere */}
      <div
        className="relative w-full max-w-sm sm:max-w-md rounded-2xl overflow-y-auto"
        style={{
          maxHeight: "92dvh",
          scrollbarWidth: "none",        /* Firefox */
          msOverflowStyle: "none",       /* IE/Edge */
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Hide scrollbar for WebKit */}
        <style>{`.result-scroll::-webkit-scrollbar { display: none; }`}</style>

        <div className="result-scroll">
          {/* Confetti lives outside the card so it fills the modal area */}
          {showConfetti && (
            <div className="absolute inset-0 pointer-events-none z-0 rounded-2xl overflow-hidden">
              <ConfettiBurst />
            </div>
          )}

          <div className="relative z-10 p-3">
            {pct < 20 ? (
              <FailedCard onClose={onClose} />
            ) : !insectData ? (
              <UnknownCard insectName={result.insect_name} confidence={result.confidence} onClose={onClose} />
            ) : (
              <>
                {pct < 60 && <UncertainBanner />}
                <InsectResultCard
                  insectData={insectData}
                  style={style}
                  confidence={result.confidence}
                  expanded={expanded}
                  onToggle={() => setExpanded((v) => !v)}
                  onClose={onClose}
                />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResultModal;