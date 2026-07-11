"use client";

import { type JobResponse } from "@/lib/api";

// Four user-facing AI steps, driven by overall progress so they advance smoothly
// regardless of which internal pipeline stages run (b-roll is often skipped).
const STEPS = [
  { title: "Analyzing content", sub: "Transcribing & understanding the video" },
  { title: "Finding highlights", sub: "Scoring the most viral-worthy moments" },
  { title: "Editing clips", sub: "Cutting vertical, face-tracked clips" },
  { title: "Adding captions", sub: "Burning captions + writing titles" },
];

function currentStep(progress: number, succeeded: boolean): number {
  if (succeeded) return STEPS.length;
  if (progress < 0.25) return 0;
  if (progress < 0.5) return 1;
  if (progress < 0.8) return 2;
  return 3;
}

export function JobProgress({ job }: { job: JobResponse }) {
  const failed = job.status === "failed";
  const succeeded = job.status === "succeeded";
  const idx = currentStep(job.progress, succeeded);
  const pct = Math.round(job.progress * 100);

  return (
    <div className="card">
      {!failed && !succeeded && (
        <p className="mb-3 text-xs text-white/40">
          Estimated time: about 2-4 minutes for most videos (longer podcasts take a bit more).
          Your first clip usually appears before the job finishes.
        </p>
      )}
      <div className="mb-5 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          {!succeeded && !failed && (
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-accent" />
            </span>
          )}
          <span className="text-sm font-medium text-white">
            {failed ? "Something went wrong" : succeeded ? "Your clips are ready" : "Creating your clips…"}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {!!job.total_clips && (job.completed_clips ?? 0) > 0 && !succeeded && (
            <span className="text-xs font-medium text-accent">
              {job.completed_clips}/{job.total_clips} clips
            </span>
          )}
          <span className="text-sm font-semibold tabular-nums text-white/70">{pct}%</span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div
          className={`h-full rounded-full transition-all duration-700 ${failed ? "bg-rose-500" : "bg-gradient-to-r from-accent/70 to-accent"}`}
          style={{ width: `${Math.max(pct, failed ? 100 : 4)}%` }}
        />
      </div>

      {/* Steps */}
      <ol className="mt-6 space-y-3">
        {STEPS.map((s, i) => {
          const done = i < idx;
          const active = i === idx && !failed;
          return (
            <li key={s.title} className="flex items-start gap-3">
              <span
                className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-xs transition ${
                  done
                    ? "border-accent bg-accent text-white"
                    : active
                    ? "border-accent text-accent"
                    : "border-white/15 text-white/30"
                }`}
              >
                {done ? "✓" : active ? <Spinner /> : i + 1}
              </span>
              <div>
                <div className={`text-sm font-medium ${done || active ? "text-white" : "text-white/35"}`}>
                  {s.title}
                </div>
                <div className={`text-xs ${active ? "text-white/50" : "text-white/25"}`}>{s.sub}</div>
              </div>
            </li>
          );
        })}
      </ol>

      {job.message && !failed && (
        <p className="mt-5 truncate text-xs text-white/35">{job.message}</p>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <svg className="h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
