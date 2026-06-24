"use client";

import { useState } from "react";
import { SignInButton, useAuth } from "@clerk/nextjs";
import {
  ApiError,
  createScript,
  type ContentType,
  type Industry,
  type ScriptLanguage,
  type ScriptResponse,
} from "@/lib/api";
import { OptionPills, type PillOption } from "@/components/studio/OptionPills";
import { GeneratingState } from "@/components/studio/GeneratingState";
import { EmptyState } from "@/components/studio/EmptyState";
import { ScriptResult } from "@/components/studio/ScriptResult";

const LANGUAGES: PillOption<ScriptLanguage>[] = [
  { value: "uz", label: "O'zbek" },
  { value: "en", label: "English" },
  { value: "ru", label: "Русский" },
];

const CONTENT_TYPES: PillOption<ContentType>[] = [
  { value: "educational", label: "Educational" },
  { value: "personal_brand", label: "Personal Brand" },
  { value: "founder_story", label: "Founder Story" },
  { value: "product_marketing", label: "Product Marketing" },
  { value: "storytelling", label: "Storytelling" },
  { value: "tutorial", label: "Tutorial" },
  { value: "sales", label: "Sales" },
  { value: "viral_reel", label: "Viral Reel" },
];

const INDUSTRIES: PillOption<Industry>[] = [
  { value: "general", label: "General" },
  { value: "business", label: "Business" },
  { value: "health", label: "Health" },
  { value: "education", label: "Education" },
  { value: "finance", label: "Finance" },
  { value: "technology", label: "Technology" },
  { value: "other", label: "Other" },
];

const DURATIONS: PillOption<"30" | "45" | "60">[] = [
  { value: "30", label: "30 sec" },
  { value: "45", label: "45 sec" },
  { value: "60", label: "60 sec" },
];

function friendlyError(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 401) return "Please sign in to generate your script.";
    if (e.status === 402) return "You're out of credits. Top up to keep creating.";
    if (e.status === 429) return "A little too fast — give it a few seconds and try again.";
    if (e.status === 502) return "The AI had trouble with that one. Tweak the topic and retry.";
    return e.message || "Something went wrong. Please try again.";
  }
  return "Network hiccup — check your connection and try again.";
}

export function ScriptStudio() {
  const { isSignedIn } = useAuth();

  const [topic, setTopic] = useState("");
  const [language, setLanguage] = useState<ScriptLanguage>("uz");
  const [contentType, setContentType] = useState<ContentType>("educational");
  const [industry, setIndustry] = useState<Industry>("general");
  const [duration, setDuration] = useState<"30" | "45" | "60">("60");

  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [result, setResult] = useState<ScriptResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function generate() {
    setStatus("loading");
    setError(null);
    setResult(null);
    try {
      const data = await createScript({
        topic: topic.trim() || undefined,
        language,
        content_type: contentType,
        industry,
        duration_seconds: parseInt(duration, 10),
      });
      setResult(data);
      setStatus("done");
    } catch (e) {
      setError(friendlyError(e));
      setStatus("error");
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      {/* Hero */}
      <div className="mb-7 text-center sm:mb-9">
        <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-4xl">
          Create your next viral video
        </h1>
        <p className="mt-2 text-sm text-white/50 sm:text-base">
          Generate scripts optimized for retention and engagement.
        </p>
      </div>

      {/* INPUT CARD */}
      <div className="card space-y-5">
        <div>
          <label htmlFor="topic" className="label">
            What's your video about?
          </label>
          <textarea
            id="topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            rows={3}
            placeholder="Example: 5 mistakes beginners make when starting a business"
            className="input resize-none text-base leading-relaxed"
          />
        </div>

        <OptionPills label="Language" options={LANGUAGES} value={language} onChange={setLanguage} />
        <OptionPills label="Content type" options={CONTENT_TYPES} value={contentType} onChange={setContentType} />

        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          <OptionPills label="Industry" options={INDUSTRIES} value={industry} onChange={setIndustry} />
          <OptionPills label="Duration" options={DURATIONS} value={duration} onChange={setDuration} />
        </div>

        {isSignedIn === false ? (
          <SignInButton mode="modal">
            <button className="btn-primary w-full py-4 text-base">Sign in to generate →</button>
          </SignInButton>
        ) : (
          <button
            type="button"
            onClick={generate}
            disabled={status === "loading"}
            className="btn-primary w-full py-4 text-base"
          >
            {status === "loading" ? "Generating…" : "✦ Generate Script"}
          </button>
        )}
      </div>

      {/* RESULT AREA */}
      <div className="mt-6">
        {status === "loading" && <GeneratingState />}

        {status === "error" && (
          <div className="card border-rose-500/30 bg-rose-500/5 text-center">
            <p className="text-sm font-medium text-rose-300">{error}</p>
            <button
              type="button"
              onClick={generate}
              className="mt-3 rounded-lg border border-white/15 px-4 py-2 text-xs font-medium text-white/80
                transition hover:border-white/30"
            >
              Try again
            </button>
          </div>
        )}

        {status === "done" && result && (
          <ScriptResult data={result} onReset={() => { setResult(null); setStatus("idle"); }} />
        )}

        {status === "idle" && <EmptyState onPick={setTopic} />}
      </div>
    </div>
  );
}
