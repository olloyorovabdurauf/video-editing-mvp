"""
Viral segment picker.

We feed GPT-4o a compressed transcript with timestamps and ask it to return
ranked candidate clips as strict JSON. The prompt is engineered around three
ideas that consistently produce hook-worthy picks:

  1. Reward openings that violate expectation ("the thing nobody tells you about X").
  2. Penalize segments that end mid-thought — chops kill watch-time on Reels.
  3. Bias toward self-contained moments (a setup + payoff inside the window).
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.reel import Segment
from app.services import ai_cache
from app.services.transcription import Transcript

SYSTEM_PROMPT = """\
You are a senior short-form video editor. Given a timestamped transcript, you
return the {n} segments most likely to perform on Reels, Shorts, and TikTok.

Hard rules:
- Each clip must be self-contained: a viewer dropping in cold understands it.
- Start on a strong hook (claim, question, or pattern interrupt) — never mid-sentence.
- End on a payoff or punchline — never mid-thought.
- Clip duration must be between 12 and {max_dur} seconds.
- Return STRICT JSON. No prose, no markdown.

Score rubric (hook_score, 0..1):
  0.9+  shocking claim, contrarian POV, or strong emotional beat
  0.7   clear insight, surprising fact
  0.5   competent but generic
  <0.5  skip
"""

USER_TEMPLATE = """\
User directive (may be empty): {prompt}

Transcript (each line = "[start-end] text"):
{lines}

Return JSON. Do NOT echo the transcript text — only timestamps + a short
reason. (We reconstruct the verbatim text ourselves from the timestamps; this
keeps the response small so it never truncates on long videos.)
{{
  "segments": [
    {{
      "start": <float>,
      "end":   <float>,
      "hook_score": <float 0..1>,
      "reason": "<one short sentence on why this hooks>"
    }}
  ]
}}
"""

# Used only when the strict pass finds nothing. A clipping tool should never
# hand the user an empty result — return the best available moment and let
# them judge. Scores will be honest (low), the duration floor is relaxed so
# short source videos still yield a clip.
LENIENT_SYSTEM_PROMPT = """\
You are a short-form video editor. The strict viral pass found nothing.
Return the SINGLE most engaging moment from this transcript anyway — even if
it is not classically viral. You MUST return exactly one segment.

Rules:
- Duration between 6 and {max_dur} seconds (or the whole clip if shorter).
- Start and end on natural sentence boundaries where possible.
- hook_score should be an honest 0..0.6 reflecting the (limited) appeal.
- Return STRICT JSON, same shape as requested.
"""


def _compress(transcript: Transcript, window_s: float = 4.0) -> str:
    """Bucket word-level timestamps into ~4s lines to keep tokens down."""
    if not transcript.words:
        return ""
    rows: list[str] = []
    buf, bucket_start = [], transcript.words[0].start
    for w in transcript.words:
        if w.end - bucket_start > window_s:
            rows.append(f"[{bucket_start:.1f}-{buf[-1].end:.1f}] {' '.join(x.text for x in buf)}")
            buf, bucket_start = [], w.start
        buf.append(w)
    if buf:
        rows.append(f"[{bucket_start:.1f}-{buf[-1].end:.1f}] {' '.join(x.text for x in buf)}")
    return "\n".join(rows)


async def pick_segments(
    transcript: Transcript,
    *,
    n: int,
    max_duration_s: int,
    prompt: str | None,
) -> list[Segment]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    compressed = _compress(transcript)
    if not compressed:
        return []

    # Cache: same video + same params → reuse picks (free + instant on retry).
    ck = ai_cache.key("segpick", compressed, n, max_duration_s, prompt or "")
    cached = ai_cache.get_json(ck)
    if cached is not None:
        logger.info("segment picks cache hit")
        return [Segment(**d) for d in cached]

    async def _ask(model: str, system: str, temperature: float, min_dur: float) -> list[Segment]:
        resp = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    prompt=prompt or "(none)", lines=compressed)},
            ],
        )
        picks = _clamp(_safe_segments(resp), transcript, max_duration_s, min_dur=min_dur)
        picks.sort(key=lambda x: x.hook_score, reverse=True)
        return picks

    strict = SYSTEM_PROMPT.format(n=n, max_dur=max_duration_s)
    lenient = LENIENT_SYSTEM_PROMPT.format(max_dur=max_duration_s)

    # Cost tiering: cheap model first; escalate to the strong model ONLY when
    # the cheap pass finds nothing. Most videos never touch the expensive model.
    out: list[Segment] = []
    try:
        logger.info("analysis (cheap={}) on {} chars", settings.openai_analysis_model, len(compressed))
        out = await _ask(settings.openai_analysis_model, strict, 0.4, min_dur=5.0)
        if not out:
            logger.info("cheap pass empty → escalating to {}", settings.openai_escalation_model)
            out = await _ask(settings.openai_escalation_model, strict, 0.4, min_dur=5.0)
        if not out:  # still nothing viral → best-available with strong model
            out = await _ask(settings.openai_escalation_model, lenient, 0.5, min_dur=4.0)
    except Exception as e:
        logger.warning("segment analysis failed: {}", e)

    # Last resort: deterministic synthesis (never return nothing for speech).
    if not out:
        synth = _synthesize_from_transcript(transcript, max_duration_s)
        out = [synth] if synth else []

    out = out[:n]
    if out:
        ai_cache.set_json(ck, [s.model_dump() for s in out], ttl_s=settings.segment_cache_ttl_s)
    return out


def _safe_segments(resp: Any) -> list[dict]:
    """
    Parse the model's JSON content defensively. response_format=json_object
    *usually* yields valid JSON, but a truncated/odd response shouldn't crash
    the whole job — return [] and let the fallback chain handle it.
    """
    content = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(content).get("segments", [])
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("segment JSON parse failed ({}); first 120 chars: {!r}",
                       e, content[:120])
        return []


def _text_in_window(transcript: Transcript, start: float, end: float) -> str:
    """Verbatim transcript text inside [start, end] — sliced from our own
    word timestamps, so it's accurate and never bloats the LLM response."""
    return " ".join(w.text for w in transcript.words if start <= w.start <= end)[:500]


def _clamp(raw: list, transcript: Transcript, max_duration_s: int, *, min_dur: float) -> list[Segment]:
    """Validate raw LLM picks (start/end/hook_score/reason) into Segments,
    filling transcript text ourselves from word timestamps."""
    out: list[Segment] = []
    for s in raw or []:
        if not isinstance(s, dict):
            continue
        try:
            start = float(s["start"]); end = float(s["end"])
            seg = Segment(
                start=start, end=end,
                hook_score=float(s.get("hook_score", 0.5)),
                reason=str(s.get("reason", ""))[:300],
                transcript=_text_in_window(transcript, start, end),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("dropping malformed segment {}: {}", s, e)
            continue
        if seg.end - seg.start < min_dur or seg.end - seg.start > max_duration_s + 5:
            continue
        out.append(seg)
    return out


def _synthesize_from_transcript(transcript: Transcript, max_duration_s: int) -> Segment | None:
    """
    Last-resort clip: cover the speech from its start up to max_duration_s.
    Honest low hook_score; the text is the verbatim words in the window.
    """
    if not transcript.words:
        return None
    start = transcript.words[0].start
    end = min(start + max_duration_s, transcript.words[-1].end)
    if end - start < 3:
        end = transcript.words[-1].end
    text = " ".join(w.text for w in transcript.words if start <= w.start <= end)
    return Segment(
        start=round(start, 2),
        end=round(end, 2),
        hook_score=0.3,
        reason="Best available moment (no strongly viral hook detected).",
        transcript=text[:500],
    )
