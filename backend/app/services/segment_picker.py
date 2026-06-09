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

Return JSON:
{{
  "segments": [
    {{
      "start": <float>,
      "end":   <float>,
      "hook_score": <float 0..1>,
      "reason": "<one sentence on why this hooks>",
      "transcript": "<verbatim text inside the window>"
    }}
  ]
}}
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

    logger.info("picking {} segments from {} chars of transcript", n, len(compressed))
    resp = await client.chat.completions.create(
        model=settings.openai_reasoning_model,
        response_format={"type": "json_object"},
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(n=n, max_dur=max_duration_s)},
            {"role": "user",   "content": USER_TEMPLATE.format(
                prompt=prompt or "(none)", lines=compressed
            )},
        ],
    )
    data: dict[str, Any] = json.loads(resp.choices[0].message.content or "{}")
    raw = data.get("segments", [])

    # Validate + clamp
    out: list[Segment] = []
    for s in raw:
        try:
            seg = Segment(**s)
        except Exception as e:
            logger.warning("dropping malformed segment {}: {}", s, e)
            continue
        if seg.end - seg.start < 5 or seg.end - seg.start > max_duration_s + 5:
            continue
        out.append(seg)
    out.sort(key=lambda x: x.hook_score, reverse=True)
    return out[:n]
