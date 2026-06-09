"use client";

import { type ReelArtifact } from "@/lib/api";

function fmt(t: number) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

export function ReelCard({ reel, index }: { reel: ReelArtifact; index: number }) {
  const score = Math.round(reel.segment.hook_score * 100);
  return (
    <div className="card overflow-hidden">
      <div className="aspect-[9/16] bg-black rounded-lg overflow-hidden mb-4">
        <video
          src={reel.output_url}
          controls
          preload="metadata"
          className="w-full h-full object-contain"
          poster={reel.thumbnail_url || undefined}
        />
      </div>
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="text-xs text-white/40 uppercase tracking-wider">Reel {index + 1}</div>
          <div className="text-sm font-mono text-white/70">
            {fmt(reel.segment.start)} → {fmt(reel.segment.end)}
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-white/40">Hook</div>
          <div
            className={`text-lg font-semibold ${
              score >= 80 ? "text-green-400" : score >= 60 ? "text-yellow-400" : "text-white/60"
            }`}
          >
            {score}
          </div>
        </div>
      </div>
      <p className="text-sm text-white/60 italic line-clamp-2">{reel.segment.reason}</p>
      <a
        href={reel.output_url}
        download
        className="mt-4 inline-block text-xs text-accent hover:underline"
      >
        Download MP4 ↓
      </a>
    </div>
  );
}
