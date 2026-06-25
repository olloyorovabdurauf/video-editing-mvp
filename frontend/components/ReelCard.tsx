"use client";

import { type ReelArtifact } from "@/lib/api";
import { CopyButton } from "@/components/CopyButton";

function fmt(t: number) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function scoreColor(score: number) {
  if (score >= 80) return "text-emerald-400 border-emerald-400/30 bg-emerald-400/10";
  if (score >= 60) return "text-amber-400 border-amber-400/30 bg-amber-400/10";
  return "text-white/60 border-white/15 bg-white/5";
}

export function ReelCard({ reel, index }: { reel: ReelArtifact; index: number }) {
  const score = Math.round((reel.segment.score ?? reel.segment.hook_score) * 100);
  const duration = Math.round(reel.segment.end - reel.segment.start);
  const title = reel.title?.trim() || `Clip ${index + 1}`;
  const captionParts = [reel.caption?.trim(), (reel.hashtags || []).map((h) => `#${h}`).join(" ")].filter(Boolean);
  const captionText = captionParts.join("\n\n");

  return (
    <div className="card group flex flex-col gap-4 overflow-hidden p-4">
      {/* Preview */}
      <div className="relative overflow-hidden rounded-xl bg-black">
        <div className="aspect-[9/16]">
          <video
            src={reel.output_url}
            controls
            preload="metadata"
            className="h-full w-full object-contain"
            poster={reel.thumbnail_url || undefined}
          />
        </div>
        {/* Viral score badge */}
        <div
          className={`absolute left-2 top-2 flex items-center gap-1 rounded-full border px-2 py-1 text-xs font-semibold backdrop-blur ${scoreColor(score)}`}
          title="Viral score — how strong the hook is"
        >
          🔥 {score}
        </div>
        {/* Duration */}
        <div className="absolute right-2 top-2 rounded-full border border-white/10 bg-black/60 px-2 py-1 font-mono text-[11px] font-semibold text-white/80 backdrop-blur">
          {duration}s
        </div>
        {/* Timestamp in original */}
        <div className="absolute bottom-2 right-2 rounded-full border border-white/10 bg-black/60 px-2 py-1 font-mono text-[11px] text-white/60 backdrop-blur">
          {fmt(reel.segment.start)}–{fmt(reel.segment.end)}
        </div>
      </div>

      {/* Title + summary + why selected */}
      <div>
        <h3 className="text-sm font-semibold leading-snug text-white">{title}</h3>
        {reel.segment.summary && (
          <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-white/60">{reel.segment.summary}</p>
        )}
        <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-white/40">
          <span className="text-white/55">Why selected:</span> {reel.segment.reason}
        </p>
      </div>

      {/* Caption */}
      {reel.caption && (
        <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[11px] uppercase tracking-wider text-white/40">Caption</span>
            <CopyButton text={captionText} />
          </div>
          <p className="line-clamp-4 whitespace-pre-line text-xs leading-relaxed text-white/70">
            {reel.caption}
          </p>
          {reel.hashtags && reel.hashtags.length > 0 && (
            <p className="mt-1.5 line-clamp-1 text-xs text-accent/80">
              {reel.hashtags.map((h) => `#${h}`).join(" ")}
            </p>
          )}
        </div>
      )}

      {/* Export */}
      <a
        href={reel.output_url}
        download
        className="mt-auto inline-flex items-center justify-center gap-1.5 rounded-lg border border-white/10
          bg-white/5 py-2.5 text-xs font-medium text-white/80 transition hover:border-accent/40 hover:text-white"
      >
        <DownloadIcon /> Download clip
      </a>
    </div>
  );
}

function DownloadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 3v12m0 0 4-4m-4 4-4-4" />
      <path d="M5 21h14" />
    </svg>
  );
}
