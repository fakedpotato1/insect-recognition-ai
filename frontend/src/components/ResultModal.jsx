import { useEffect, useRef } from "react";
import { X, CheckCircle, AlertTriangle, XCircle } from "lucide-react";

// ─── Confetti ────────────────────────────────────────────────────────────────
function ConfettiBurst() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const pieces = Array.from({ length: 120 }, () => ({
      x: canvas.width / 2,
      y: canvas.height / 2,
      r: Math.random() * 6 + 3,
      color: ["#0e7490", "#06b6d4", "#a5f3fc", "#166534", "#bbf7d0", "#fbbf24", "#f87171"][
        Math.floor(Math.random() * 7)
      ],
      angle: Math.random() * 2 * Math.PI,
      speed: Math.random() * 7 + 3,
      gravity: 0.18,
      vx: 0,
      vy: 0,
      alpha: 1,
      shape: Math.random() > 0.5 ? "rect" : "circle",
    }));

    pieces.forEach((p) => {
      p.vx = Math.cos(p.angle) * p.speed;
      p.vy = Math.sin(p.angle) * p.speed - 4;
    });

    let frame;
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      pieces.forEach((p) => {
        p.vy += p.gravity;
        p.x += p.vx;
        p.y += p.vy;
        p.alpha -= 0.012;
        if (p.alpha <= 0) return;
        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = p.color;
        if (p.shape === "rect") {
          ctx.fillRect(p.x - p.r / 2, p.y - p.r / 2, p.r, p.r * 1.6);
        } else {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.r / 2, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      ctx.globalAlpha = 1;
      if (pieces.some((p) => p.alpha > 0)) {
        frame = requestAnimationFrame(animate);
      }
    };
    animate();
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none rounded-2xl"
    />
  );
}

// ─── Confidence config ────────────────────────────────────────────────────────
function getConfidenceConfig(confidence) {
  const pct = confidence * 100;
  if (pct < 20) return null; // handled outside
  if (pct < 60) {
    return {
      level: "Medium",
      barColor: "bg-yellow-400",
      textColor: "text-yellow-600",
      bgColor: "bg-yellow-50",
      borderColor: "border-yellow-200",
      icon: <AlertTriangle size={20} className="text-yellow-500" />,
      badge: (
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-yellow-100 text-yellow-700 text-xs font-semibold">
          <AlertTriangle size={12} /> Uncertain
        </span>
      ),
      caution: true,
    };
  }
  return {
    level: "High",
    barColor: "bg-green-500",
    textColor: "text-green-600",
    bgColor: "bg-cyan-50",
    borderColor: "border-cyan-200",
    icon: <CheckCircle size={20} className="text-green-500" />,
    badge: (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-green-100 text-green-700 text-xs font-semibold">
        <CheckCircle size={12} /> Confident
      </span>
    ),
    caution: false,
    confetti: true,
  };
}

// ─── Progress Bar ─────────────────────────────────────────────────────────────
function ConfidenceBar({ confidence, config }) {
  const pct = (confidence * 100).toFixed(1);
  return (
    <div className="mt-4 w-full">
      <div className="flex items-center justify-between mb-1">
        <span className={`text-xs font-semibold uppercase tracking-wide ${config.textColor}`}>
          Confidence: {config.level}
        </span>
        {config.badge}
      </div>
      {/* Bar with tooltip on hover */}
      <div className="relative group w-full h-3 bg-slate-100 rounded-full overflow-visible">
        <div
          className={`h-3 rounded-full transition-all duration-700 ${config.barColor}`}
          style={{ width: `${pct}%` }}
        />
        {/* Hover tooltip */}
        <div className="absolute -top-8 left-1/2 -translate-x-1/2 hidden group-hover:flex items-center px-2 py-1 bg-slate-800 text-white text-xs rounded-lg whitespace-nowrap shadow-lg z-10">
          {pct}%
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800" />
        </div>
      </div>
    </div>
  );
}

// ─── Main Modal ───────────────────────────────────────────────────────────────
function ResultModal({ result, onClose }) {
  if (!result) return null;

  const pct = result.confidence * 100;

  // ── Detection failed (< 20%) ──
  if (pct < 20) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4 animate-fade-in">
        <div className="relative w-full max-w-md bg-white rounded-2xl shadow-2xl p-8 text-center animate-zoom-in">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 h-8 w-8 flex items-center justify-center rounded-full bg-slate-100 hover:bg-slate-200 text-slate-500 transition"
          >
            <X size={16} />
          </button>
          <div className="flex justify-center mb-4">
            <div className="h-16 w-16 rounded-full bg-red-100 flex items-center justify-center">
              <XCircle size={36} className="text-red-500" />
            </div>
          </div>
          <h2 className="text-xl font-bold text-slate-800 mb-2">Detection Failed</h2>
          <p className="text-slate-500 text-sm">
            We couldn't confidently identify the insect. Please try again with a
            clearer, well-lit photo where the insect is fully visible.
          </p>
          <button
            onClick={onClose}
            className="mt-6 w-full py-2.5 bg-red-500 hover:bg-red-600 text-white font-medium rounded-lg transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  const config = getConfidenceConfig(result.confidence);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4 animate-fade-in">
      <div className={`relative w-full max-w-md bg-white rounded-2xl shadow-2xl p-8 text-center border ${config.borderColor} animate-zoom-in overflow-hidden`}>

        {/* Confetti (high confidence only) */}
        {config.confetti && <ConfettiBurst />}

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 h-8 w-8 flex items-center justify-center rounded-full bg-slate-100 hover:bg-slate-200 text-slate-500 transition z-10"
        >
          <X size={16} />
        </button>

        {/* Icon */}
        <div className="flex justify-center mb-4 relative z-10">
          <div className={`h-16 w-16 rounded-full ${config.bgColor} flex items-center justify-center`}>
            {config.icon}
          </div>
        </div>

        {/* Label */}
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-1 relative z-10">
          Detection Result
        </p>

        {/* Insect name */}
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-800 mb-3 relative z-10">
          {result.insect_name}
        </h2>

        {/* Confidence bar */}
        <div className="relative z-10">
          <ConfidenceBar confidence={result.confidence} config={config} />
        </div>

        {/* Caution message for uncertain results */}
        {config.caution && (
          <p className="mt-4 text-xs text-yellow-700 bg-yellow-50 border border-yellow-200 rounded-lg px-3 py-2 relative z-10">
            ⚠️ This result has low confidence. Please use with caution and consider retaking with a clearer image.
          </p>
        )}

        {/* Close button at bottom */}
        <button
          onClick={onClose}
          className="mt-6 w-full py-2.5 bg-cyan-700 hover:bg-cyan-800 text-cyan-50 font-medium rounded-lg transition-colors relative z-10"
        >
          Close
        </button>
      </div>
    </div>
  );
}

export default ResultModal;