"use client";

import { useEffect, useState } from "react";

// AI-style staged progress — communicates the model is *working through* the
// script, not a generic spinner. Purely cosmetic; the real result arrives async.
const STAGES = [
  "Analyzing your topic…",
  "Engineering the 5-second hook…",
  "Structuring the retention arc…",
  "Writing the value section…",
  "Closing the loop + CTA…",
  "Polishing the caption…",
];

export function GeneratingState() {
  const [i, setI] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setI((p) => (p + 1) % STAGES.length), 1400);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="card relative overflow-hidden">
      {/* sweeping sheen */}
      <div className="pointer-events-none absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-accent/10 to-transparent" />
      <div className="flex items-center gap-3">
        <span className="relative flex h-3 w-3">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
          <span className="relative inline-flex h-3 w-3 rounded-full bg-accent" />
        </span>
        <span className="text-sm font-medium text-white">Creating your script…</span>
      </div>

      <p className="mt-2 text-sm text-white/50 transition-all">{STAGES[i]}</p>

      {/* indeterminate bar */}
      <div className="mt-5 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div className="h-full w-1/3 animate-[slide_1.5s_ease-in-out_infinite] rounded-full bg-gradient-to-r from-accent/40 via-accent to-accent/40" />
      </div>

      {/* skeleton preview lines */}
      <div className="mt-6 space-y-3">
        {[90, 75, 82, 60].map((w, k) => (
          <div
            key={k}
            className="h-3 animate-pulse rounded bg-white/5"
            style={{ width: `${w}%`, animationDelay: `${k * 120}ms` }}
          />
        ))}
      </div>
    </div>
  );
}
