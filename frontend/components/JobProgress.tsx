"use client";

import { type JobResponse, type ReelStatus } from "@/lib/api";

const STAGES: { key: ReelStatus; label: string }[] = [
  { key: "queued",            label: "Queued" },
  { key: "downloading",       label: "Downloading" },
  { key: "transcribing",      label: "Transcribing" },
  { key: "analyzing",         label: "Picking segments" },
  { key: "generating_broll",  label: "Generating b-roll" },
  { key: "rendering",         label: "Rendering" },
  { key: "succeeded",         label: "Done" },
];

export function JobProgress({ job }: { job: JobResponse }) {
  const currentIdx = Math.max(0, STAGES.findIndex(s => s.key === job.status));
  const pct = Math.round(job.progress * 100);

  return (
    <div className="card">
      <div className="flex items-baseline justify-between mb-4">
        <div>
          <div className="text-sm text-white/50">Job</div>
          <div className="font-mono text-sm">{job.job_id}</div>
        </div>
        <div className="text-right">
          <div className="text-sm text-white/50">Progress</div>
          <div className="text-2xl font-semibold tabular-nums">{pct}%</div>
        </div>
      </div>

      <div className="h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-500 ${
            job.status === "failed" ? "bg-red-500" : "bg-accent"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* On narrow screens: vertical list with full labels (no overlap). */}
      {/* On sm+: horizontal pill row, labels under the chips. */}
      <ol className="mt-6 sm:hidden space-y-2">
        {STAGES.map((s, i) => {
          const done = i < currentIdx || job.status === "succeeded";
          const active = i === currentIdx && job.status !== "succeeded" && job.status !== "failed";
          return (
            <li key={s.key} className="flex items-center gap-3">
              <div
                className={`w-6 h-6 shrink-0 rounded-full border-2 flex items-center justify-center text-xs ${
                  done
                    ? "bg-accent border-accent text-white"
                    : active
                    ? "border-accent text-accent animate-pulse"
                    : "border-white/15 text-white/30"
                }`}
              >
                {done ? "✓" : i + 1}
              </div>
              <span
                className={`text-sm ${done || active ? "text-white/90" : "text-white/30"}`}
              >
                {s.label}
              </span>
            </li>
          );
        })}
      </ol>

      <ol className="mt-6 hidden sm:grid sm:grid-cols-7 gap-2">
        {STAGES.map((s, i) => {
          const done = i < currentIdx || job.status === "succeeded";
          const active = i === currentIdx && job.status !== "succeeded" && job.status !== "failed";
          return (
            <li key={s.key} className="text-center">
              <div
                className={`w-7 h-7 mx-auto rounded-full border-2 flex items-center justify-center text-xs ${
                  done
                    ? "bg-accent border-accent text-white"
                    : active
                    ? "border-accent text-accent animate-pulse"
                    : "border-white/15 text-white/30"
                }`}
              >
                {done ? "✓" : i + 1}
              </div>
              <div
                className={`mt-2 text-[11px] leading-tight ${
                  done || active ? "text-white/80" : "text-white/30"
                }`}
              >
                {s.label}
              </div>
            </li>
          );
        })}
      </ol>

      {job.message && (
        <div className="mt-4 text-sm text-white/60 italic">{job.message}</div>
      )}
    </div>
  );
}
