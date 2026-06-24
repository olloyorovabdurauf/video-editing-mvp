"use client";

import type { ScriptResponse, ScriptSection } from "@/lib/api";
import { CopyButton } from "@/components/CopyButton";

const BEAT_LABEL: Record<ScriptSection["name"], string> = {
  hook: "Hook",
  problem: "Problem",
  value: "Value",
  payoff: "Payoff + CTA",
};

const BEAT_DOT: Record<ScriptSection["name"], string> = {
  hook: "bg-accent",
  problem: "bg-amber-400",
  value: "bg-emerald-400",
  payoff: "bg-violet-400",
};

function captionText(s: ScriptResponse): string {
  const tags = s.hashtags.map((h) => (h.startsWith("#") ? h : `#${h}`)).join(" ");
  return [s.caption.hook, s.caption.body, s.caption.cta, tags].filter(Boolean).join("\n\n");
}

export function ScriptResult({ data, onReset }: { data: ScriptResponse; onReset: () => void }) {
  const caption = captionText(data);

  return (
    <div className="space-y-4">
      {/* Title + global actions */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <span className="label mb-1">Title</span>
          <h2 className="text-xl font-semibold leading-snug text-white sm:text-2xl">{data.title}</h2>
          <div className="mt-2 flex flex-wrap gap-1.5 text-[11px] text-white/40">
            <Tag>{data.duration_seconds}s</Tag>
            <Tag>{data.content_type.replace(/_/g, " ")}</Tag>
            <Tag>{data.industry}</Tag>
            <Tag>{data.language.toUpperCase()}</Tag>
          </div>
        </div>
        <div className="flex gap-2">
          <CopyButton text={data.formatted} label="Copy all" />
          <button
            type="button"
            onClick={onReset}
            className="rounded-lg border border-white/10 px-3 py-2 text-xs font-medium text-white/70
              transition hover:border-white/20 hover:text-white active:scale-[0.97]"
          >
            ＋ New script
          </button>
        </div>
      </div>

      {/* HOOK — large highlighted */}
      <div className="card relative overflow-hidden bg-gradient-to-br from-accent/15 to-transparent">
        <div className="flex items-center justify-between">
          <span className="label mb-0 text-accent/80">Hook · 0-5s</span>
          <CopyButton text={data.hook} />
        </div>
        <p className="mt-3 text-lg font-semibold leading-snug text-white sm:text-xl">{data.hook}</p>
      </div>

      {/* SCRIPT — timeline */}
      <div className="card">
        <div className="mb-4 flex items-center justify-between">
          <span className="label mb-0">Script</span>
          <CopyButton text={data.script} label="Copy script" />
        </div>
        <ol className="space-y-5">
          {data.sections.map((s) => (
            <li key={s.name} className="relative pl-6">
              <span
                className={`absolute left-0 top-1.5 h-2.5 w-2.5 rounded-full ${BEAT_DOT[s.name]}`}
                aria-hidden
              />
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-xs text-white/40">{s.time_range}</span>
                <span className="text-xs font-semibold uppercase tracking-wider text-white/55">
                  {BEAT_LABEL[s.name]}
                </span>
              </div>
              <p className="mt-1.5 text-[15px] leading-relaxed text-white/90">{s.voiceover}</p>
              {s.visual && (
                <p className="mt-1 text-xs text-white/35">
                  <span className="text-white/50">On-screen:</span> {s.visual}
                </p>
              )}
            </li>
          ))}
        </ol>
      </div>

      {/* CAPTION — ready to copy */}
      <div className="card">
        <div className="mb-3 flex items-center justify-between">
          <span className="label mb-0">Caption</span>
          <CopyButton text={caption} label="Copy caption" />
        </div>
        <p className="font-medium text-white">{data.caption.hook}</p>
        <p className="mt-2 whitespace-pre-line text-sm leading-relaxed text-white/70">{data.caption.body}</p>
        <p className="mt-2 text-sm font-medium text-accent">{data.caption.cta}</p>
        {data.hashtags.length > 0 && (
          <p className="mt-3 text-sm text-white/40">
            {data.hashtags.map((h) => (h.startsWith("#") ? h : `#${h}`)).join(" ")}
          </p>
        )}
      </div>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-md border border-white/10 bg-white/[0.03] px-1.5 py-0.5 capitalize">
      {children}
    </span>
  );
}
