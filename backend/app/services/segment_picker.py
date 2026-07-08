"""
Complete-clip picker.

This is a long-form → short-form *repurposing* step, NOT a highlight extractor.
We ask the model for SELF-CONTAINED clips (a full hook → context → value →
payoff arc) sized for Reels/Shorts/TikTok, and then we mechanically enforce the
rules the model can't be trusted to follow exactly:

  1. Duration is SEMANTIC: [min_duration_s, max_duration_s] (default 45-60s) is
     a soft preference — the clip ends where the idea ends (42s complete beats
     60s padded; 67s to finish a thought is allowed). A hard 0.6x-1.5x band
     (capped at 90s) still rejects fragments and the whole-video bug.
  2. Start/end are snapped to SENTENCE boundaries so a clip never opens or
     closes mid-thought.
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

# Overall score = a viral-editor blend of the full rubric. Structure
# (completeness) dominates so we never reward a lone hook; retention + virality
# capture shareability. standalone_score is applied as a hard GATE in _ask, not a
# weight. Tunable; must sum to 1.0.
_WEIGHTS = {
    "hook_score":         0.13,
    "curiosity_score":    0.09,
    "emotion_score":      0.10,
    "value_score":        0.14,
    "retention_score":    0.12,
    "completeness_score": 0.18,
    "payoff_score":       0.14,   # the conclusion / resolution
    "virality_score":     0.10,
}


def _overall(scores: dict) -> float:
    """Weighted overall quality in [0,1] from the rubric dimensions."""
    return round(sum(w * float(scores.get(k, 0.0)) for k, w in _WEIGHTS.items()), 3)

SYSTEM_PROMPT = """\
You are a TOP 1% short-form video editor (think Opus Clip / a senior TikTok
editor) turning a long video into COMPLETE, standalone clips for Reels, Shorts
and TikTok. You are NOT a highlight extractor: never return a lone hook, a clever
sentence, or a contextless moment.

WORK IN TWO STEPS.

STEP 1 — UNDERSTAND THE WHOLE VIDEO. Read the entire transcript and build a
semantic map: identify the major topics, stories, lessons, arguments, emotional
moments and transitions. Split it into LOGICAL STORY BLOCKS by MEANING, never by
time — each block is one complete unit of communication with a real beginning and
end. Put that map in "story_map".

STEP 2 — SELECT {n} CLIPS from those blocks, spread across DIFFERENT parts of the
video (beginning, middle, end) — not {n} variations of the same moment. Each clip
MUST:
- AIM for {min_dur}-{max_dur} seconds — but MEANING BEATS DURATION: end the
  clip where the idea naturally finishes. A complete idea at 40s beats a
  padded 45s one; an idea that needs ~70s to reach its conclusion may take
  it. Never pad, never truncate a thought to hit the target.
- Follow a COMPLETE arc the viewer understands WITHOUT the original video:
    0-5s            POWER HOOK — a line that opens a curiosity gap or a stake
    5-15s           CONTEXT — what is going on, so the topic is instantly clear
    15-{value_end}s BODY — the full value/story/explanation, no missing logic
    {value_end}-end CONCLUSION — a clear lesson, insight, payoff or takeaway
- Begin at the START of a thought (a sentence boundary), never mid-sentence.
- End on a FINISHED thought — never mid-sentence, never on "...and that's why".

PREFER clips containing: a surprising fact, a strong opinion or controversy, an
emotional beat, a transformation, a mistake and its lesson, a story, a curiosity
gap, or concrete actionable advice. AVOID generic filler and throat-clearing.

REJECT a clip if it: starts in the middle, ends abruptly, has a weak or missing
conclusion, or lacks the context needed to make sense on its own. Fix it first —
move the start earlier to capture the setup, the end later to capture the payoff —
or discard it.

THE STANDALONE TEST (apply to EVERY clip): "If someone watched ONLY this clip on
TikTok, with no other context, would they fully understand it and feel it was
complete?" If NO, do not return it.

Score every clip 0..1:
  hook_score          strength of the first 3-5s
  curiosity_score     how strong the opening curiosity gap / open question is
  emotion_score       emotional pull (surprise, inspiration, tension, humor)
  value_score         how much real insight/value the body delivers
  retention_score     will a viewer keep watching to the end (pacing, payoff tease)
  completeness_score  full beginning -> middle -> end arc. MOST IMPORTANT.
  payoff_score        how resolved/satisfying the ending (the conclusion) is
  standalone_score    understandable with ZERO external context (the test above)
  virality_score      shareability / likelihood to perform on short-form
Only return clips with completeness_score >= 0.6 AND standalone_score >= 0.6. If
the video has fewer than {n} genuinely complete clips, return FEWER — never pad
with fragments.

Return STRICT JSON. No prose, no markdown.
"""

USER_TEMPLATE = """\
User directive (may be empty): {prompt}

Transcript (each line = "[start-end] text"):
{lines}

First map the video into story blocks, then pick complete {min_dur}-{max_dur}s
clips from them. Do NOT echo the transcript text — only timestamps, scores, a
one-line reason and a one-line summary (we reconstruct the verbatim text from the
timestamps).
{{
  "story_map": [
    {{"start": <float>, "end": <float>, "theme": "<short topic>",
      "type": "story|lesson|argument|emotional|tip|other"}}
  ],
  "segments": [
    {{
      "start": <float, a sentence boundary>,
      "end": <float, a sentence boundary, {min_dur}-{max_dur}s after start>,
      "hook_score": <0..1>, "curiosity_score": <0..1>, "emotion_score": <0..1>,
      "value_score": <0..1>, "retention_score": <0..1>,
      "completeness_score": <0..1>, "payoff_score": <0..1>,
      "standalone_score": <0..1>, "virality_score": <0..1>,
      "reason": "<why this works as a complete standalone clip>",
      "summary": "<the complete idea this clip explains, one line>"
    }}
  ]
}}
"""

# Fill pass: we already have some clips but fewer than the user asked for. Find
# MORE distinct complete clips from OTHER story blocks, relaxing the bar slightly
# (still full {min_dur}-{max_dur}s, sentence-bounded, standalone).
FILL_SYSTEM_PROMPT = """\
You are a top short-form video editor. Find up to {n} MORE distinct, self-
contained clips (~{min_dur}-{max_dur}s preferred, completeness first) from story blocks not already obviously
covered. Each must still be at least {min_dur}s, begin and end on sentence
boundaries, cover a WHOLE thought (hook -> context -> body -> conclusion) and pass
the standalone test — never a fragment. These can be solid rather than perfect:
score completeness/standalone honestly (0.45-0.8). The
0-5s/5-15s/15-{value_end}s/{value_end}-end structure still applies. Same JSON
shape (story_map + segments with the full score set). STRICT JSON.
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

    # v4: invalidates clips cached before the hierarchical rubric (new score
    # dimensions + standalone gate change which clips win).
    ck = ai_cache.key("segpick_v6", compressed, n, min_eff, max_duration_s, prompt or "")
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
            max_tokens=2600,               # room for the story_map + richer scores
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": USER_TEMPLATE.format(
                    prompt=prompt or "(none)", lines=compressed,
                    min_dur=min_eff, max_dur=max_duration_s)},
            ],
        )
        picks = _clamp(_safe_segments(resp), transcript, min_eff, max_duration_s)
        # Two gates: a complete arc AND it stands alone with zero external context
        # (Issue 4). Both track the pass's bar so the fill pass can relax together.
        picks = [p for p in picks
                 if p.completeness_score >= min_completeness
                 and p.standalone_score >= min_completeness]
        picks.sort(key=lambda x: x.score, reverse=True)
        return picks

    strict = SYSTEM_PROMPT.format(n=n, min_dur=min_eff, max_dur=max_duration_s, value_end=value_end)
    fill = FILL_SYSTEM_PROMPT.format(n=n, min_dur=min_eff, max_dur=max_duration_s, value_end=value_end)

    out: list[Segment] = []
    try:
        logger.info("clip analysis (cheap={}) target n={} {}-{}s on {} chars",
                    settings.openai_analysis_model, n, min_eff, max_duration_s, len(compressed))
        out = await _ask(settings.openai_analysis_model, strict, 0.4, min_completeness=0.5)
        # Escalate whenever we're SHORT of the requested count — not only at zero.
        # This is the "user asked for 5, got 1" fix: keep gathering distinct clips
        # until we reach n (or the video genuinely runs out of complete material).
        if len(_dedupe_overlap(out)) < n:
            logger.info("cheap pass gave {}/{} → escalating to {}",
                        len(out), n, settings.openai_escalation_model)
            out = _merge(out, await _ask(settings.openai_escalation_model, strict, 0.5, min_completeness=0.5))
        if len(_dedupe_overlap(out)) < n:
            logger.info("still short ({}/{}) → relaxed fill pass", len(_dedupe_overlap(out)), n)
            out = _merge(out, await _ask(settings.openai_escalation_model, fill, 0.6, min_completeness=0.4))
    except Exception as e:
        logger.warning("segment analysis failed: {}", e)

    out = _dedupe_overlap(out)                 # drop near-duplicate / overlapping windows
    # Guarantee the requested count on a long-enough video: if the AI under-
    # delivered, take complete 45-60s windows from evenly-spaced regions it didn't
    # already cover. A human editor does the same — one from each part of the talk.
    if len(out) < n:
        logger.info("AI gave {}/{} → distributing clips across the timeline", len(out), n)
        out = _dedupe_overlap(_distribute_clips(transcript, out, n, min_eff, max_duration_s))
    out.sort(key=lambda x: x.score, reverse=True)
    out = out[:n]
    out.sort(key=lambda x: x.start)            # chronological order for the user

    if not out:
        synth = _synthesize_from_transcript(transcript, min_eff, max_duration_s)
        out = [synth] if synth else []

    # AGENT 4 — Quality Reviewer: re-read each FINALIZED clip's exact words
    # (not the compressed buckets it was picked from) and veto/nudge before
    # anything renders. This is the "would a cold viewer understand?" gate.
    try:
        out = await _review_clips(client, transcript, out, min_eff, max_duration_s)
    except Exception as e:
        logger.warning("quality reviewer failed (keeping picks): {}", e)

    if out:
        ai_cache.set_json(ck, [s.model_dump() for s in out], ttl_s=settings.segment_cache_ttl_s)
    return out


_REVIEW_PROMPT = """You are a ruthless short-form QUALITY REVIEWER. For each numbered clip you get
its EXACT spoken words, plus PRE/POST — the words just BEFORE and AFTER the clip
(NOT included in it). Validate every clip with these six questions:
1. Does the viewer immediately understand the topic?
2. Is enough context provided?
3. Is the main idea fully explained?
4. Is there a complete conclusion?
5. Does the ending feel natural (not mid-explanation, not abrupt)?
6. Does it stand alone without the original video?
THE FINISH GATE (apply before anything else): "Has the speaker FINISHED the
idea?" If the final words leave the thought hanging and the conclusion is
visible in POST, EXTEND end_delta to include it — keep extending until the
payoff sentence is inside the clip. A viewer must never feel "wait... what
happened next?".
Every clip must contain ALL FIVE: (1) hook, (2) context, (3) main idea,
(4) evidence/example, (5) conclusion/payoff. Missing parts that exist in
PRE/POST → pull them in with boundary moves; truly absent → reject.
Fix boundaries with start_delta/end_delta seconds (-30..30; negative
start_delta pulls PRE words in, positive end_delta pulls POST words in; 0 if
fine).
DURATION IS NOT A RULE: ~45-60s is preferred, but completeness always wins — a
complete 42s idea is right, an idea needing 67s to finish is right. Never
shorten a clip if that removes context or the conclusion; never stop before an
explanation, argument or example is finished.
Return STRICT JSON: {"reviews": [{"i": <n>, "verdict": "approve"|"reject",
"start_delta": <s>, "end_delta": <s>, "reason": "<short>"}]}.
Reject ONLY when some answer stays NO even after boundary moves. Approve good
clips — don't nitpick."""


async def _review_clips(client, transcript: Transcript, segs: list[Segment],
                        min_s: int, max_s: int) -> list[Segment]:
    if not segs:
        return segs
    settings = get_settings()

    def words_in(a: float, b: float) -> str:
        return " ".join(w.text for w in transcript.words if a <= w.start < b)

    async def review_round(cands: list[Segment]) -> tuple[list[Segment], int]:
        body = "\n\n".join(
            f"CLIP {i} [{s.start:.0f}s-{s.end:.0f}s]\n"
            f"PRE: {words_in(max(0.0, s.start - 15), s.start)}\n"
            f"TEXT: {words_in(s.start, s.end)}\n"
            f"POST: {words_in(s.end, s.end + 15)}"
            for i, s in enumerate(cands))
        resp = await client.chat.completions.create(
            model=settings.openai_analysis_model,
            response_format={"type": "json_object"}, temperature=0.2, max_tokens=900,
            messages=[{"role": "system", "content": _REVIEW_PROMPT},
                      {"role": "user", "content": body}])
        reviews = {int(r.get("i", -1)): r
                   for r in json.loads(resp.choices[0].message.content or "{}").get("reviews", [])}
        approved: list[Segment] = []
        rejects = 0
        for i, s in enumerate(cands):
            r = reviews.get(i)
            if r is None:
                approved.append(s)               # no verdict → don't lose the clip
                continue
            if r.get("verdict") == "reject":
                rejects += 1
                logger.info("reviewer rejected clip {} ({}s-{}s): {}",
                            i, round(s.start), round(s.end), r.get("reason", ""))
                continue
            ds = max(-30.0, min(30.0, float(r.get("start_delta") or 0)))
            de = max(-30.0, min(30.0, float(r.get("end_delta") or 0)))
            if ds or de:                         # nudge, then re-align to clean cuts
                win = _finalize_window(transcript.words, s.start + ds, s.end + de, min_s, max_s)
                if win:
                    s = s.model_copy(update={"start": win[0], "end": win[1]})
            approved.append(s)
        return approved, rejects

    want = len(segs)
    kept, rejects = await review_round(segs)

    # "Choose another story": rejects left us short → pick complete windows from
    # story regions not already covered, and put THOSE through review too.
    if rejects and len(kept) < want:
        pool = _dedupe_overlap(_distribute_clips(transcript, kept, want, min_s, max_s))
        fresh = [s for s in pool if all(abs(s.start - k.start) > 1.0 for k in kept)][: want - len(kept)]
        if fresh:
            logger.info("reviewer: sourcing {} replacement clip(s) from uncovered stories", len(fresh))
            approved2, _ = await review_round(fresh)
            kept = _dedupe_overlap(kept + approved2)

    if not kept:                                 # never 0 clips for a video with speech
        logger.warning("reviewer rejected ALL clips — keeping top-scored as fallback")
        kept = [max(segs, key=lambda x: x.score)]
    kept.sort(key=lambda x: x.start)
    return kept[:want]


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


# Semantic-boundary bands. [min_s, max_s] is a SOFT preference (the product's
# 45-60s target); completeness outranks it. The HARD band is the sanity/platform
# envelope: fragments below it are rejected, and nothing may exceed it (Reels/
# Shorts cap out around 90s — and it's the whole-video-bug guard).
_HARD_MIN_FRACTION = 0.6      # e.g. min 45 → never below 27s
_HARD_MAX_FRACTION = 1.5      # e.g. max 60 → never above 90s
_PLATFORM_CAP_S = 90.0


def _finalize_window(words: list[Word], start: float, end: float,
                     min_s: float, max_s: float) -> tuple[float, float] | None:
    """
    Align [start, end] to where the IDEA begins and naturally finishes.

    The requested `end` is treated as where the thought ends — the cut lands on
    the sentence end nearest to it. [min_s, max_s] is only a soft preference:
    a 42s complete idea is not stretched, a 67s one is not truncated. The hard
    band (fractions above, capped at _PLATFORM_CAP_S) bounds the result so we
    never emit fragments or the whole video. Prefers SENTENCE boundaries, falls
    back to WORD boundaries (Whisper often omits punctuation), then a time cut.
    Returns None only for an essentially-empty transcript or an unusable window.
    """
    if len(words) < 2:
        return None
    t0, t1 = words[0].start, words[-1].end
    total = t1 - t0
    eff_min = min(min_s, total)                  # source shorter than the floor → whole thing
    hard_min = min(max(15.0, _HARD_MIN_FRACTION * min_s), total, eff_min)
    hard_max = min(_PLATFORM_CAP_S, _HARD_MAX_FRACTION * max_s)

    start = max(t0, min(start, t1))
    sent_starts, sent_ends = _boundaries(words)
    word_starts = [w.start for w in words]
    word_ends = [w.end for w in words]

    # Snap START to the nearest clean boundary at/just before the requested start.
    # Prefer a sentence start, but only if one began RECENTLY (within max_s) —
    # otherwise a single trivial sentence-start at t=0 (no punctuation in the
    # transcript) would drag every window back to the beginning. Else use the
    # nearest word start.
    sent_cand = [s for s in sent_starts if start - max_s <= s <= start + 0.4]
    word_cand = [w for w in word_starts if w <= start + 0.2]
    cand = sent_cand or word_cand
    start = max(cand) if cand else t0

    # Where the idea ends, per the selection/review stages — honored even below
    # min_s (short-but-complete beats padded). A request beyond the hard band is
    # NOT an idea's end (it's the whole-video failure mode) — aim at the soft
    # target instead of stretching to the cap.
    ideal = end if (end - start) <= hard_max else start + max_s
    target_end = min(t1, max(ideal, start + hard_min))

    def _pick_end(boundaries: list[float]) -> float | None:
        usable = [b for b in boundaries if hard_min <= (b - start) <= hard_max]
        if not usable:
            under = [b for b in boundaries if start < b <= start + hard_max]
            return max(under) if under else None
        # MEANING FIRST: the sentence end nearest to where the idea actually
        # ends — even when that's past max_s. (A previous version preferred any
        # end inside the soft band, which cut 67s ideas at ~58s "to stay under
        # 60" — exactly the interrupted-speaker feel we must never ship.)
        return min(usable, key=lambda b: abs(b - target_end))

    end = _pick_end(sent_ends) if sent_ends else None
    if end is None:
        end = _pick_end(word_ends)
    if end is None:
        end = min(start + max_s, t1)

    end = min(end, t1, start + hard_max)         # HARD CAP — never the whole video
    if end - start < hard_min - 2.0:             # unusable fragment
        end = min(t1, start + max_s)

    if end - start < min(hard_min, total) - 2.0:
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
        scores = {
            "hook_score":         _f(s.get("hook_score")),
            "curiosity_score":    _f(s.get("curiosity_score")),
            "emotion_score":      _f(s.get("emotion_score")),
            "value_score":        _f(s.get("value_score")),
            "retention_score":    _f(s.get("retention_score")),
            "completeness_score": _f(s.get("completeness_score")),
            "payoff_score":       _f(s.get("payoff_score")),
            "standalone_score":   _f(s.get("standalone_score")),
            "virality_score":     _f(s.get("virality_score")),
        }
        out.append(Segment(
            start=ws, end=we,
            **scores, score=_overall(scores),
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
        # Degenerate transcript — take from the start but HARD-CAP at max_s so we
        # never emit the whole video as a single clip (the 895s-clip bug).
        ws = words[0].start
        we = min(words[-1].end, ws + max_s)
    else:
        ws, we = win
    text = _text_in_window(transcript, ws, we)
    scores = {"hook_score": 0.4, "curiosity_score": 0.4, "emotion_score": 0.4,
              "value_score": 0.5, "retention_score": 0.45, "completeness_score": 0.6,
              "payoff_score": 0.4, "standalone_score": 0.55, "virality_score": 0.4}
    return Segment(
        start=round(ws, 2), end=round(we, 2),
        **scores, score=_overall(scores),
        reason="Best available complete section (no strongly viral hook detected).",
        summary=(text[:120] + "…") if len(text) > 120 else text,
        transcript=text,
    )


def _distribute_clips(transcript: Transcript, have: list[Segment], n: int,
                      min_s: int, max_s: int) -> list[Segment]:
    """
    Fill up to `n` clips by taking a complete 45-60s window from evenly-spaced
    regions of the video that aren't already covered. Guarantees the requested
    count on a long-enough source — a human editor pulls one clip from each part
    of a talk rather than returning a single highlight.
    """
    words = transcript.words
    if not words or len(have) >= n:
        return have
    t0, t1 = words[0].start, words[-1].end
    total = t1 - t0
    if total < min_s:                       # too short to add anything meaningful
        return have

    kept = list(have)
    step = total / n
    for k in range(n):
        if len(kept) >= n:
            break
        rstart = t0 + k * step
        win = _finalize_window(words, rstart, rstart + max_s, float(min_s), float(max_s))
        if win is None:
            continue
        ws, we = win
        overlaps = any(
            (min(we, c.end) - max(ws, c.start)) > 0.4 * min(we - ws, c.end - c.start)
            for c in kept
        )
        if overlaps:
            continue
        text = _text_in_window(transcript, ws, we)
        scores = {"hook_score": 0.45, "curiosity_score": 0.4, "emotion_score": 0.4,
                  "value_score": 0.5, "retention_score": 0.45, "completeness_score": 0.55,
                  "payoff_score": 0.45, "standalone_score": 0.5, "virality_score": 0.4}
        kept.append(Segment(
            start=round(ws, 2), end=round(we, 2),
            **scores, score=_overall(scores),
            reason="A complete section from this part of the video.",
            summary=(text[:120] + "…") if len(text) > 120 else text,
            transcript=text,
        ))
    return kept
