"use client";

import { useState } from "react";

import { sendFeedback, type FeedbackReason } from "@/lib/api";

const REASONS: { key: FeedbackReason; label: string }[] = [
  { key: "story_incomplete", label: "Story incomplete" },
  { key: "weak_hook", label: "Weak hook" },
  { key: "wrong_crop", label: "Wrong crop" },
  { key: "subtitle_issue", label: "Subtitle issue" },
  { key: "wrong_language", label: "Wrong language" },
  { key: "not_viral", label: "Not viral enough" },
  { key: "render_quality", label: "Render quality" },
  { key: "other", label: "Other" },
];

/** 👍/👎 per clip; 👎 expands into one-tap reasons. Fire-and-forget. */
export function ClipFeedback({ jobId, clipIndex }: { jobId: string; clipIndex: number }) {
  const [state, setState] = useState<"idle" | "picking" | "done">("idle");

  const send = (verdict: "up" | "down", reason?: FeedbackReason) => {
    sendFeedback(jobId, clipIndex, verdict, reason).catch(() => {});
    setState("done");
  };

  if (state === "done") {
    return <p className="mt-2 text-xs text-white/40">Thanks — this improves future clips.</p>;
  }
  if (state === "picking") {
    return (
      <div className="mt-2 flex flex-wrap gap-1.5">
        {REASONS.map((r) => (
          <button
            key={r.key}
            type="button"
            onClick={() => send("down", r.key)}
            className="rounded-full border border-white/10 px-2.5 py-1 text-[11px] text-white/60 transition hover:border-white/30 hover:text-white"
          >
            {r.label}
          </button>
        ))}
      </div>
    );
  }
  return (
    <div className="mt-2 flex items-center gap-2 text-xs text-white/40">
      <span>Rate this clip</span>
      <button type="button" aria-label="Great clip" onClick={() => send("up")}
              className="rounded-md px-1.5 py-0.5 transition hover:bg-white/10">👍</button>
      <button type="button" aria-label="Needs improvement" onClick={() => setState("picking")}
              className="rounded-md px-1.5 py-0.5 transition hover:bg-white/10">👎</button>
    </div>
  );
}
