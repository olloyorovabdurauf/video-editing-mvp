"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createReel, type AspectRatio, type CaptionStyle } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [sourceUrl, setSourceUrl] = useState("");
  const [aspect, setAspect] = useState<AspectRatio>("9:16");
  const [count, setCount] = useState(3);
  const [maxDur, setMaxDur] = useState(45);
  const [prompt, setPrompt] = useState("");
  const [captionStyle, setCaptionStyle] = useState<CaptionStyle>("karaoke");
  const [smartCrop, setSmartCrop] = useState(true);
  const [useAi, setUseAi] = useState(false);   // premium opt-in
  const [budget, setBudget] = useState(4);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const job = await createReel({
        source_url: sourceUrl,
        aspect,
        target_count: count,
        max_duration_s: maxDur,
        caption_style: captionStyle,
        smart_crop: smartCrop,
        add_broll: true,
        add_music: true,
        use_ai_broll: useAi,
        ai_broll_budget_usd: budget,
        prompt: prompt || undefined,
      });
      router.push(`/jobs/${job.job_id}`);
    } catch (err: any) {
      setError(err.message || "Submission failed");
      setSubmitting(false);
    }
  }

  return (
    <div className="grid lg:grid-cols-5 gap-8">
      <section className="lg:col-span-3">
        <h1 className="text-3xl font-semibold tracking-tight mb-2">
          Turn a long video into reels — automatically.
        </h1>
        <p className="text-white/60 mb-8 leading-relaxed">
          Paste a YouTube URL. We transcribe it, find the most hook-worthy
          moments, generate cinematic AI b-roll, sync captions, and ship MP4s.
        </p>

        <form onSubmit={onSubmit} className="card space-y-6">
          <div>
            <label className="label">Source video</label>
            <input
              required
              type="url"
              className="input"
              placeholder="https://youtube.com/watch?v=..."
              value={sourceUrl}
              onChange={(e) => setSourceUrl(e.target.value)}
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="label">Aspect</label>
              <select
                className="input"
                value={aspect}
                onChange={(e) => setAspect(e.target.value as AspectRatio)}
              >
                <option value="9:16">9:16 — Reels / Shorts</option>
                <option value="16:9">16:9 — YouTube</option>
                <option value="1:1">1:1 — Square</option>
              </select>
            </div>
            <div>
              <label className="label">Reels to extract</label>
              <input
                type="number"
                min={1}
                max={10}
                className="input"
                value={count}
                onChange={(e) => setCount(parseInt(e.target.value) || 1)}
              />
            </div>
            <div>
              <label className="label">Max length (s)</label>
              <input
                type="number"
                min={10}
                max={180}
                className="input"
                value={maxDur}
                onChange={(e) => setMaxDur(parseInt(e.target.value) || 45)}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Caption style</label>
              <select
                className="input"
                value={captionStyle}
                onChange={(e) => setCaptionStyle(e.target.value as CaptionStyle)}
              >
                <option value="karaoke">Karaoke — word lights up as spoken</option>
                <option value="popup">Pop-up — one word at a time</option>
                <option value="minimal">Minimal — clean &amp; subtle</option>
                <option value="none">None</option>
              </select>
            </div>
            <div className="flex items-end">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={smartCrop}
                  onChange={(e) => setSmartCrop(e.target.checked)}
                  className="accent-accent"
                />
                <span>
                  <span className="font-medium">Smart crop</span>
                  <span className="block text-white/40 text-xs">
                    Face-tracking pan &amp; scan (vs center)
                  </span>
                </span>
              </label>
            </div>
          </div>

          <div>
            <label className="label">Creative directive (optional)</label>
            <textarea
              className="input min-h-[80px]"
              placeholder="focus on contrarian moments; keep energy high; cut around any filler words"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>

          <div className="border-t border-white/5 pt-6">
            <div className="flex items-start gap-3 mb-4">
              <input
                id="ai"
                type="checkbox"
                checked={useAi}
                onChange={(e) => setUseAi(e.target.checked)}
                className="mt-1 accent-accent"
              />
              <label htmlFor="ai" className="text-sm">
                <span className="font-medium">
                  Generate AI b-roll
                  <span className="ml-2 px-2 py-0.5 text-[10px] rounded-full bg-accent/20 text-accent uppercase tracking-wider">
                    Premium
                  </span>
                </span>
                <span className="block text-white/50">
                  Cinematic, custom-prompted shots. ~$0.50/clip. Off by default;
                  stock b-roll covers most cases for free.
                </span>
              </label>
            </div>
            {useAi && (
              <div>
                <label className="label">Budget cap (USD)</label>
                <input
                  type="number"
                  min={0}
                  max={20}
                  step={0.5}
                  className="input"
                  value={budget}
                  onChange={(e) => setBudget(parseFloat(e.target.value) || 0)}
                />
              </div>
            )}
          </div>

          {error && (
            <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3">
              {error}
            </div>
          )}

          <button type="submit" className="btn-primary w-full" disabled={submitting}>
            {submitting ? "Submitting…" : "Generate reels"}
          </button>
        </form>
      </section>

      <aside className="lg:col-span-2 space-y-4">
        <div className="card">
          <h2 className="font-medium mb-3">The pipeline</h2>
          <ol className="text-sm text-white/70 space-y-2">
            <li><span className="text-accent">1.</span> Download &amp; transcribe (Whisper)</li>
            <li><span className="text-accent">2.</span> Pick viral segments (GPT-4o)</li>
            <li><span className="text-accent">3.</span> Plan b-roll insertions</li>
            <li><span className="text-accent">4.</span> Generate b-roll (Runway / Higgsfield)</li>
            <li><span className="text-accent">5.</span> Reframe, composite, caption, mix</li>
          </ol>
        </div>
        <div className="card text-xs text-white/50 leading-relaxed">
          Generation runs in parallel across all insertion points.
          Typical job: <span className="text-white/80">2–4 min</span> end to end,
          dominated by AI b-roll inference (~60–120s/clip).
        </div>
      </aside>
    </div>
  );
}
