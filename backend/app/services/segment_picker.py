"""
Complete-clip picker.

This is a long-form → short-form *repurposing* step, NOT a highlight extractor.
We ask the model for SELF-CONTAINED clips (a full hook → context → value →
payoff arc) sized for Reels/Shorts/TikTok, and then we mechanically enforce the
rules the model can't be trusted to follow exactly:

  1. Duration is forced into [min_duration_s, max_duration_s] (default 45-60s) —
     never a 5-10s fragment.
  2. Start/end are snapped to SENTENCE boundaries, and the window is extended to
     reach the minimum length, so a clip never opens or closes mid-thought.
  3. Clips are ranked by an overall score that weights *story completeness* most,
     not the hook alone. Clips below a completeness threshold are dropped.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.reel import Segment
from app.services import ai_cache
from app.services.transcription import Transcript, Word

# Overall score weights. Completeness + value dominate so we stop rewarding lone
# hooks. Tunable, must sum to 1.0.
_W_HOOK, _W_VALUE, _W_COMPLETE, _W_PAYOFF = 0.20, 0.25, 0.35, 0.20

SYSTEM_PROMPT = """\
You are a professional short-form video editor turning a long video into
COMPLETE standalone clips for Reels, Shorts and TikTok. You are NOT a highlight
extractor: never return a lone hook, a clever sentence, or a contextless moment.

Find {n} DISTINCT complete clips spread across DIFFERENT parts of the video
(beginning, middle, end) — not {n} variations of the same moment. Aim for exactly
{n}. Each clip MUST:
- Be between {min_dur} and {max_dur} seconds long. NEVER shorter than {min_dur}s.
- Contain a COMPLETE idea the viewer fully understands WITHOUT the original video:
    0-5s            a strong hook / clear opening
    5-15s           context or the problem
    15-{value_end}s the main value, story, or explanation
    {value_end}-end a conclusion / payoff that resolves the opening
- Begin at the START of a thought (a sentence boundary), never mid-sentence.
- End on a finished thought — never mid-sentence, never on "...and that's why".

Do NOT select on emotional words, controversy, or energy alone. Ask: does this
section have a clear beginning, explain a full idea, deliver value, and end
satisfyingly? Could it stand alone as a Reel?

For EACH clip score 0..1:
  hook_score          strength of the opening
  value_score         how much insight/value the body delivers
  completeness_score  does it stand alone as a full idea with a clear beginning,
                      middle and end? THIS MATTERS MOST.
  payoff_score        how resolved/satisfying the ending is
Only return clips with completeness_score >= 0.6. If the video has fewer than
{n} genuinely complete clips, return FEWER — never pad with fragments.

Return STRICT JSON. No prose, no markdown.
"""

USER_TEMPLATE = """\
User directive (may be empty): {prompt}

Transcript (each line = "[start-end] text"):
{lines}

Pick complete {min_dur}-{max_dur}s clips. Return JSON. Do NOT echo the transcript
text — only timestamps, scores, a one-line reason and a one-line summary (we
reconstruct the verbatim text ourselves from the timestamps).
{{
  "segments": [
    {{
      "start": <float, a sentence boundary>,
      "end": <float, a sentence boundary, {min_dur}-{max_dur}s after start>,
      "hook_score": <0..1>, "value_score": <0..1>,
      "completeness_score": <0..1>, "payoff_score": <0..1>,
      "reason": "<why this works as a complete standalone clip>",
      "summary": "<the complete idea this clip explains, one line>"
    }}
  ]
}}
"""

# Fill pass: we already have some clips but fewer than the user asked for. Find
# MORE distinct complete clips from OTHER parts of the video, relaxing the
# completeness bar slightly (still full {min_dur}-{max_dur}s, sentence-bounded).
FILL_SYSTEM_PROMPT = """\
You are a short-form video editor. Find up to {n} MORE distinct, self-contained
{min_dur}-{max_dur}s clips from parts of the video not already obviously covered.
Each must still be at least {min_dur}s, begin and end on sentence boundaries, and
cover a WHOLE thought (hook → context → value → payoff) — never a fragment. These
can be solid rather than perfect: score completeness honestly (0.45-0.8). The
0-5s/5-15s/15-{value_end}s/{value_end}-end structure still applies. STRICT JSON,
same shape as requested.
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
    min_duration_s: int,
    max_duration_s: int,
    prompt: str | None,
) -> list[Segment]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    compressed = _compress(transcript)
    if not compressed:
        return []
    # Source shorter than the floor → the whole thing is the only "clip".
    total = transcript.words[-1].end - transcript.words[0].start
    min_eff = min(min_duration_s, int(total))

    ck = ai_cache.key("segpick_v2", compressed, n, min_eff, max_duration_s, prompt or "")
    cached = ai_cache.get_json(ck)
    if cached is not None:
        logger.info("segment picks cache hit")
        return [Segment(**d) for d in cached]

    value_end = max(min_eff, max_duration_s - 15)

    async def _ask(model: str, system: str, temperature: float, min_completeness: float) -> list[Segment]:
        resp = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=2200,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": USER_TEMPLATE.format(
                    prompt=prompt or "(none)", lines=compressed,
                    min_dur=min_eff, max_dur=max_duration_s)},
            ],
        )
        picks = _clamp(_safe_segments(resp), transcript, min_eff, max_duration_s)
        picks = [p for p in picks if p.completeness_score >= min_completeness]
        picks.sort(key=lambda x: x.score, reverse=True)
        return picks

    strict = SYSTEM_PROMPT.format(n=n, min_dur=min_eff, max_dur=max_duration_s, value_end=value_end)
    fill = FILL_SYSTEM_PROMPT.format(n=n, min_dur=min_eff, max_dur=max_duration_s, value_end=value_end)

    out: list[Segment] = []
    try:
        logger.info("clip analysis (cheap={}) target n={} {}-{}s on {} chars",
                    settings.openai_analysis_model, n, min_eff, max_duration_s, len(compressed))
        out = await _ask(settings.openai_analysis_model, strict, 0.4, min_completeness=0.6)
        # Escalate whenever we're SHORT of the requested count — not only at zero.
        # This is the "user asked for 5, got 1" fix: keep gathering distinct clips
        # until we reach n (or the video genuinely runs out of complete material).
        if len(_dedupe_overlap(out)) < n:
            logger.info("cheap pass gave {}/{} → escalating to {}",
                        len(out), n, settings.openai_escalation_model)
            out = _merge(out, await _ask(settings.openai_escalation_model, strict, 0.5, min_completeness=0.6))
        if len(_dedupe_overlap(out)) < n:
            logger.info("still short ({}/{}) → relaxed fill pass", len(_dedupe_overlap(out)), n)
            out = _merge(out, await _ask(settings.openai_escalation_model, fill, 0.6, min_completeness=0.45))
    except Exception as e:
        logger.warning("segment analysis failed: {}", e)

    out = _dedupe_overlap(out)                 # drop near-duplicate / overlapping windows
    out.sort(key=lambda x: x.score, reverse=True)
    out = out[:n]
    out.sort(key=lambda x: x.start)            # chronological order for the user

    if not out:
        synth = _synthesize_from_transcript(transcript, min_eff, max_duration_s)
        out = [synth] if synth else []

    if out:
        ai_cache.set_json(ck, [s.model_dump() for s in out], ttl_s=settings.segment_cache_ttl_s)
    return out


def _merge(a: list[Segment], b: list[Segment]) -> list[Segment]:
    return list(a) + list(b)


def _dedupe_overlap(segs: list[Segment], max_overlap: float = 0.4) -> list[Segment]:
    """
    Keep distinct clips: greedily accept the highest-scoring, rejecting any clip
    that overlaps an already-kept one by more than `max_overlap` of the shorter
    clip. Prevents three "clips" that are really the same moment.
    """
    kept: list[Segment] = []
    for s in sorted(segs, key=lambda x: x.score, reverse=True):
        clash = False
        for k in kept:
            inter = min(s.end, k.end) - max(s.start, k.start)
            if inter > 0:
                shorter = min(s.end - s.start, k.end - k.start) or 1.0
                if inter / shorter > max_overlap:
                    clash = True
                    break
        if not clash:
            kept.append(s)
    return kept


def _safe_segments(resp: Any) -> list[dict]:
    content = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(content).get("segments", [])
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("segment JSON parse failed ({}); first 120 chars: {!r}", e, content[:120])
        return []


def _f(v: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Sentence-boundary editing — so clips never open or close mid-thought.
# ---------------------------------------------------------------------------

def _is_sentence_end(text: str) -> bool:
    # A word ends a sentence if its last char is terminal punctuation. Decimals
    # like "3.5" end in a digit (not a period), so no special-casing is needed.
    t = text.rstrip()
    return bool(t) and t[-1] in ".!?…"


def _boundaries(words: list[Word]) -> tuple[list[float], list[float]]:
    """Times where sentences START and END, from word punctuation."""
    starts: list[float] = []
    ends: list[float] = []
    new_sentence = True
    for w in words:
        if new_sentence:
            starts.append(w.start)
        if _is_sentence_end(w.text):
            ends.append(w.end)
            new_sentence = True
        else:
            new_sentence = False
    return starts, ends


def _finalize_window(words: list[Word], start: float, end: float,
                     min_s: float, max_s: float) -> tuple[float, float] | None:
    """
    Snap [start, end] to sentence boundaries and force the duration into
    [min_s, max_s] by extending to the nearest sentence ends. Returns None if a
    minimum-length window can't be formed (source genuinely too short here).
    """
    if len(words) < 2:
        return None
    t0, t1 = words[0].start, words[-1].end
    starts, ends = _boundaries(words)
    if not starts or not ends:
        return None

    start = max(t0, min(start, t1))
    # Snap start back to the sentence start at/just before the requested start.
    cand = [s for s in starts if s <= start + 0.4]
    start = max(cand) if cand else t0

    target_end = max(end, start + min_s)
    # Prefer a sentence-end that lands the duration inside [min_s, max_s].
    in_range = [e for e in ends if min_s <= (e - start) <= max_s]
    if in_range:
        # closest sentence-end to the model's requested end
        end = min(in_range, key=lambda e: abs(e - target_end))
    else:
        under = [e for e in ends if (e - start) <= max_s and e > start]
        end = max(under) if under else min(start + max_s, t1)

    end = min(end, t1, start + max_s)
    dur = end - start
    if dur < min(min_s, t1 - t0) - 2.0:
        return None
    return (round(start, 2), round(end, 2))


def _text_in_window(transcript: Transcript, start: float, end: float) -> str:
    return " ".join(w.text for w in transcript.words if start <= w.start <= end)[:1500]


def _clamp(raw: list, transcript: Transcript, min_s: int, max_s: int) -> list[Segment]:
    """Validate raw picks into complete, boundary-aligned, duration-correct clips."""
    words = transcript.words
    out: list[Segment] = []
    for s in raw or []:
        if not isinstance(s, dict):
            continue
        try:
            start = float(s["start"]); end = float(s["end"])
        except (KeyError, ValueError, TypeError):
            continue
        win = _finalize_window(words, start, end, float(min_s), float(max_s))
        if win is None:
            continue
        ws, we = win
        hook = _f(s.get("hook_score")); value = _f(s.get("value_score"))
        comp = _f(s.get("completeness_score")); pay = _f(s.get("payoff_score"))
        overall = round(_W_HOOK * hook + _W_VALUE * value
                        + _W_COMPLETE * comp + _W_PAYOFF * pay, 3)
        out.append(Segment(
            start=ws, end=we,
            hook_score=hook, value_score=value, completeness_score=comp,
            payoff_score=pay, score=overall,
            reason=str(s.get("reason", ""))[:300],
            summary=str(s.get("summary", ""))[:300],
            transcript=_text_in_window(transcript, ws, we),
        ))
    return out


def _synthesize_from_transcript(transcript: Transcript, min_s: int, max_s: int) -> Segment | None:
    """
    Last resort: a complete-length window starting at the first sentence, snapped
    to sentence boundaries. Honest mid scores — better than an empty result.
    """
    words = transcript.words
    if not words:
        return None
    win = _finalize_window(words, words[0].start, words[0].start + max_s, float(min_s), float(max_s))
    if win is None:
        ws, we = words[0].start, words[-1].end           # whole (short) clip
    else:
        ws, we = win
    text = _text_in_window(transcript, ws, we)
    return Segment(
        start=round(ws, 2), end=round(we, 2),
        hook_score=0.4, value_score=0.5, completeness_score=0.6, payoff_score=0.4,
        score=round(_W_HOOK * 0.4 + _W_VALUE * 0.5 + _W_COMPLETE * 0.6 + _W_PAYOFF * 0.4, 3),
        reason="Best available complete section (no strongly viral hook detected).",
        summary=(text[:120] + "…") if len(text) > 120 else text,
        transcript=text,
    )
