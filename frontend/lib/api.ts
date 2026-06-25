// Thin typed wrapper over the FastAPI surface. Keep this file the single
// place we declare backend response shapes — keep them in sync with
// backend/app/schemas/reel.py.

export type AspectRatio = "9:16" | "16:9" | "1:1";

export type ReelStatus =
  | "queued"
  | "downloading"
  | "transcribing"
  | "analyzing"
  | "generating_broll"
  | "rendering"
  | "succeeded"
  | "failed";

export type CaptionStyle = "karaoke" | "popup" | "minimal" | "none";

export interface ReelCreateRequest {
  source_url: string;
  aspect: AspectRatio;
  target_count: number;
  max_duration_s: number;
  min_duration_s: number;
  caption_style: CaptionStyle;
  smart_crop: boolean;
  add_broll: boolean;            // stock (Pexels) — free
  add_music: boolean;
  use_ai_broll: boolean;         // premium tier — costs credits
  ai_broll_budget_usd: number;
  prompt?: string;
  user_id?: string;
}

export interface Segment {
  start: number;
  end: number;
  hook_score: number;
  value_score?: number;
  completeness_score?: number;
  payoff_score?: number;
  score?: number;        // overall, completeness-weighted
  reason: string;
  summary?: string;
  transcript: string;
}

export interface ReelArtifact {
  segment: Segment;
  output_url: string;
  thumbnail_url?: string | null;
  title?: string;
  caption?: string;
  hashtags?: string[];
}

export interface JobResponse {
  job_id: string;
  status: ReelStatus;
  progress: number;
  message?: string | null;
  total_clips?: number | null;
  completed_clips?: number | null;
  artifacts: ReelArtifact[];
}

// Demo mode: when NEXT_PUBLIC_DEMO=1 we skip the backend and serve a scripted
// job lifecycle. Useful for UI screenshots, design review, and the case where
// you want to show the product without firing up Celery + Whisper.
const DEMO = process.env.NEXT_PUBLIC_DEMO === "1";

// Bearer-token getter. In prod this is replaced by Clerk's `await getToken()`.
// Pluggable so we can wire any auth provider without touching call sites.
type TokenGetter = () => Promise<string | null>;
let _tokenGetter: TokenGetter = async () => null;
export function setTokenGetter(fn: TokenGetter) {
  _tokenGetter = fn;
}

async function authHeaders(): Promise<Record<string, string>> {
  const tok = await _tokenGetter();
  return tok ? { Authorization: `Bearer ${tok}` } : {};
}

const DEMO_TIMELINE: { status: ReelStatus; progress: number; message: string; delay: number }[] = [
  { status: "queued",            progress: 0.02, message: "queued", delay: 0 },
  { status: "downloading",       progress: 0.10, message: "downloading source from YouTube", delay: 1500 },
  { status: "transcribing",      progress: 0.25, message: "Whisper transcribing 23m of audio", delay: 3500 },
  { status: "analyzing",         progress: 0.42, message: "GPT-4o picking viral segments", delay: 6000 },
  { status: "generating_broll",  progress: 0.55, message: "planning AI b-roll insertions", delay: 8500 },
  { status: "generating_broll",  progress: 0.65, message: "generating 4 AI b-roll clip(s)", delay: 11000 },
  { status: "rendering",         progress: 0.78, message: "rendering clip 1/3 — smart crop", delay: 14000 },
  { status: "rendering",         progress: 0.88, message: "rendering clip 2/3 — captions + music", delay: 16500 },
  { status: "succeeded",         progress: 1.00, message: "3 reels ready", delay: 19000 },
];

const DEMO_ARTIFACTS: ReelArtifact[] = [
  {
    segment: { start: 127.5, end: 168.0, hook_score: 0.92,
               reason: "Contrarian POV on remote work — opens with a claim that violates expectation.",
               transcript: "Most people think productivity is about doing more, but it's actually about saying no." },
    output_url: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
  },
  {
    segment: { start: 412.0, end: 449.0, hook_score: 0.81,
               reason: "Memorable analogy with strong visual payoff.",
               transcript: "Meetings are like printers — everyone needs one, nobody wants to own it." },
    output_url: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
  },
  {
    segment: { start: 890.0, end: 921.0, hook_score: 0.74,
               reason: "Specific story with concrete numbers — high credibility signal.",
               transcript: "We cut our standup from 30 minutes to 6 and shipped 40% faster the next quarter." },
    output_url: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
  },
];

const DEMO_START_KEY = "demo_started_at";

export async function createReel(body: ReelCreateRequest): Promise<JobResponse> {
  if (DEMO) {
    const id = `demo-${Date.now().toString(36)}`;
    sessionStorage.setItem(`${DEMO_START_KEY}:${id}`, String(Date.now()));
    return { job_id: id, status: "queued", progress: 0, message: "demo", artifacts: [] };
  }
  const r = await fetch("/api/v1/reels", {
    method: "POST",
    headers: { "content-type": "application/json", ...(await authHeaders()) },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`createReel failed: ${r.status} ${await r.text()}`);
  return r.json();
}

// ---------------------------------------------------------------------------
// Script generation (text product). Keep shapes in sync with
// backend/app/schemas/script.py.
// ---------------------------------------------------------------------------

export type ScriptLanguage = "uz" | "en" | "ru";
export type ContentType =
  | "educational"
  | "personal_brand"
  | "founder_story"
  | "product_marketing"
  | "storytelling"
  | "tutorial"
  | "sales"
  | "viral_reel";
export type Industry =
  | "general" | "business" | "health" | "education" | "finance" | "technology" | "other";

export interface ScriptGenerateRequest {
  topic?: string;
  language: ScriptLanguage;
  content_type: ContentType;
  industry: Industry;
  duration_seconds: number;
}

export interface ScriptSection {
  name: "hook" | "problem" | "value" | "payoff";
  start_s: number;
  end_s: number;
  voiceover: string;
  visual: string;
  time_range: string;
}

export interface ScriptCaption {
  hook: string;
  body: string;
  cta: string;
}

export interface ScriptResponse {
  title: string;
  hook: string;
  script: string;
  sections: ScriptSection[];
  caption: ScriptCaption;
  hashtags: string[];
  language: ScriptLanguage;
  content_type: ContentType;
  industry: Industry;
  duration_seconds: number;
  formatted: string;
}

export async function createScript(body: ScriptGenerateRequest): Promise<ScriptResponse> {
  const r = await fetch("/api/v1/scripts", {
    method: "POST",
    headers: { "content-type": "application/json", ...(await authHeaders()) },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = "";
    try {
      detail = (await r.json())?.detail ?? "";
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(r.status, detail || `Request failed (${r.status})`);
  }
  return r.json();
}

export interface Balance {
  user_id: string;
  balance: number;
}

export async function getBalance(): Promise<Balance> {
  const r = await fetch("/api/v1/billing/balance", {
    cache: "no-store",
    headers: await authHeaders(),
  });
  if (!r.ok) throw new ApiError(r.status, `balance failed (${r.status})`);
  return r.json();
}

/** Typed error so the UI can show friendly, status-aware messages. */
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export async function getJob(id: string): Promise<JobResponse> {
  if (DEMO) {
    let startedAt = parseInt(sessionStorage.getItem(`${DEMO_START_KEY}:${id}`) || "0");
    if (!startedAt) {
      startedAt = Date.now();
      sessionStorage.setItem(`${DEMO_START_KEY}:${id}`, String(startedAt));
    }
    const elapsed = Date.now() - startedAt;
    const stage = [...DEMO_TIMELINE].reverse().find((s) => elapsed >= s.delay) ?? DEMO_TIMELINE[0];
    return {
      job_id: id,
      status: stage.status,
      progress: stage.progress,
      message: stage.message,
      artifacts: stage.status === "succeeded" ? DEMO_ARTIFACTS : [],
    };
  }
  const r = await fetch(`/api/v1/reels/${id}`, {
    cache: "no-store",
    headers: await authHeaders(),
  });
  if (!r.ok) throw new Error(`getJob failed: ${r.status}`);
  return r.json();
}
