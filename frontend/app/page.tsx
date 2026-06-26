"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { SignInButton, useAuth } from "@clerk/nextjs";
import { ApiError, createReel } from "@/lib/api";

const CLIP_COUNTS = [3, 5, 8];

const LANGUAGES = [
  { value: "auto", label: "Auto-detect" },
  { value: "uz", label: "O'zbek" },
  { value: "en", label: "English" },
  { value: "ru", label: "Русский" },
  { value: "kk", label: "Қазақша" },
  { value: "ar", label: "العربية" },
];

const HOW_IT_WORKS = [
  { icon: "🎧", title: "Analyzes", desc: "Transcribes & understands the whole video" },
  { icon: "✨", title: "Finds highlights", desc: "Scores the most viral-worthy moments" },
  { icon: "✂️", title: "Edits clips", desc: "Cuts vertical, face-tracked short clips" },
  { icon: "💬", title: "Captions", desc: "Burns captions + writes titles & captions" },
];

function friendlyError(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 401) return "Please sign in to create clips.";
    if (e.status === 402) return "You're out of credits. Top up to keep creating.";
    if (e.status === 429) return "A little too fast — give it a moment and try again.";
    if (e.status === 400) return "That link doesn't look right. Paste a public video URL.";
    return e.message || "Something went wrong. Please try again.";
  }
  return "Network hiccup — check your connection and try again.";
}

export default function HomePage() {
  const router = useRouter();
  const { isSignedIn } = useAuth();

  const [url, setUrl] = useState("");
  const [count, setCount] = useState(3);
  const [language, setLanguage] = useState("auto");
  const [captions, setCaptions] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const looksValid = /^https?:\/\/.+\..+/.test(url.trim());

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!looksValid) {
      setError("Paste a full video link (e.g. https://youtube.com/watch?v=…).");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const job = await createReel({
        source_url: url.trim(),
        aspect: "9:16",
        target_count: count,
        min_duration_s: 45,
        max_duration_s: 60,
        language: language === "auto" ? undefined : language,
        caption_style: captions ? "karaoke" : "none",
        smart_crop: true,
        add_broll: true,
        add_music: true,
        use_ai_broll: false,
        ai_broll_budget_usd: 0,
      });
      router.push(`/jobs/${job.job_id}`);
    } catch (err) {
      setError(friendlyError(err));
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      {/* Hero */}
      <div className="mb-8 text-center sm:mb-10">
        <span className="inline-block rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-white/50">
          AI video repurposing
        </span>
        <h1 className="mt-4 text-3xl font-semibold tracking-tight text-white sm:text-5xl">
          Turn long videos into{" "}
          <span className="bg-gradient-to-r from-accent to-violet-300 bg-clip-text text-transparent">
            viral short clips
          </span>
        </h1>
        <p className="mx-auto mt-3 max-w-lg text-sm text-white/50 sm:text-base">
          Paste a YouTube link. Our AI finds the best moments and edits them into
          ready-to-post clips — captions, titles and all.
        </p>
      </div>

      {/* Input card */}
      <form onSubmit={onSubmit} className="card space-y-5">
        <div>
          <label htmlFor="url" className="label">
            YouTube video URL
          </label>
          <div className="relative">
            <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-white/30">
              <YoutubeIcon />
            </span>
            <input
              id="url"
              type="url"
              inputMode="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://youtube.com/watch?v=…"
              className="input py-4 pl-11 text-base"
            />
          </div>
        </div>

        {/* Minimal options */}
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <div>
            <span className="label">Language</span>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="input h-9 w-40 py-0 text-sm"
              title="Captions always match this. Pick your video's language for best accuracy."
            >
              {LANGUAGES.map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <span className="label">Clips</span>
            <div className="flex gap-2">
              {CLIP_COUNTS.map((n) => (
                <button
                  key={n}
                  type="button"
                  onClick={() => setCount(n)}
                  className={`h-9 w-11 rounded-lg border text-sm font-medium transition active:scale-95 ${
                    count === n
                      ? "border-accent/60 bg-accent/15 text-white"
                      : "border-white/10 bg-white/[0.03] text-white/60 hover:text-white"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
          <label className="flex h-9 cursor-pointer select-none items-center gap-2 text-sm text-white/70">
            <input
              type="checkbox"
              checked={captions}
              onChange={(e) => setCaptions(e.target.checked)}
              className="h-4 w-4 accent-accent"
            />
            Burn captions
          </label>
        </div>

        {error && <p className="text-sm text-rose-300">{error}</p>}

        {isSignedIn === false ? (
          <SignInButton mode="modal">
            <button type="button" className="btn-primary w-full py-4 text-base">
              Sign in to create clips →
            </button>
          </SignInButton>
        ) : (
          <button
            type="submit"
            disabled={submitting || !looksValid}
            className="btn-primary w-full py-4 text-base"
          >
            {submitting ? "Starting…" : "✦ Create Clips"}
          </button>
        )}
        <p className="text-center text-xs text-white/30">
          Works with podcasts, interviews, talks and any long-form video.
        </p>
      </form>

      {/* How it works */}
      <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {HOW_IT_WORKS.map((s) => (
          <div key={s.title} className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
            <div className="text-xl">{s.icon}</div>
            <div className="mt-2 text-sm font-medium text-white">{s.title}</div>
            <div className="mt-0.5 text-xs leading-snug text-white/40">{s.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function YoutubeIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M23 12s0-3.2-.4-4.7a2.5 2.5 0 0 0-1.8-1.8C19.3 5 12 5 12 5s-7.3 0-8.8.5A2.5 2.5 0 0 0 1.4 7.3C1 8.8 1 12 1 12s0 3.2.4 4.7a2.5 2.5 0 0 0 1.8 1.8C4.7 19 12 19 12 19s7.3 0 8.8-.5a2.5 2.5 0 0 0 1.8-1.8C23 15.2 23 12 23 12ZM9.8 15.3V8.7l6.2 3.3-6.2 3.3Z" />
    </svg>
  );
}
