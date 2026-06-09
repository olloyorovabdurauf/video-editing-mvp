"""
Scene extraction: WHERE inside a segment should b-roll appear?

Not every moment benefits from b-roll:
  - The first 2-3 seconds (the "hook") should stay on the speaker — viewer
    needs to lock onto a face.
  - The closing 1-2 seconds (the payoff / CTA) usually return to speaker.
  - Mid-segment exposition, metaphors, and abstractions are the sweet spot.
  - Strong emotional beats *on* the speaker — leave alone.

We give GPT-4o the segment's word-timestamped transcript and ask it to
return a list of "InsertionWindow" objects. The orchestrator then compiles
a prompt for each window and submits to the generator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.reel import Segment


@dataclass
class InsertionWindow:
    start: float            # seconds, ABSOLUTE in the source video
    end: float
    rationale: str          # for debugging / UX explanation
    transcript: str         # the words inside the window
    priority: float = 0.5   # 0..1; we drop lowest-priority if budget runs out


SCENE_SYSTEM = """\
You decide where in a short-form video clip we should overlay b-roll.

Hard rules:
- Never overlay the first 2.5 seconds of the clip (hook reads the speaker's face).
- Never overlay the last 1.5 seconds (payoff/CTA back on speaker).
- B-roll windows must be between 2.5s and 5.0s long.
- B-roll windows must NOT overlap each other.
- Leave at least 1.0s of speaker-on-camera between windows.
- Prefer to overlay during: exposition, metaphor, lists, abstract claims,
  setting the scene, naming a place/object/concept that has a clear visual.
- Avoid overlaying during: direct address to viewer, emotional reactions,
  one-liner jokes, names of specific people.
- Return STRICT JSON. No prose, no markdown.
"""

SCENE_USER = """\
Clip spans absolute timestamps [{seg_start:.2f} .. {seg_end:.2f}] in the source video.

Transcript (lines = "[absolute_start-absolute_end] words"):
{lines}

Return JSON:
{{
  "windows": [
    {{
      "start": <float, ABSOLUTE timestamp>,
      "end":   <float, ABSOLUTE timestamp>,
      "rationale": "<one short sentence>",
      "transcript": "<verbatim words inside the window>",
      "priority":  <float 0..1, higher = more important>
    }}
  ]
}}

If no good moment exists, return {{"windows": []}}.
"""


async def extract_scenes(
    segment: Segment,
    *,
    max_windows: int = 3,
) -> list[InsertionWindow]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # We re-bucket the segment transcript into ~1s lines for the prompt.
    # Caller already has the bucketed lines from segment_picker, but we need
    # absolute timestamps so we accept just the segment transcript string here
    # and let the LLM reason about boundaries.
    duration = segment.end - segment.start
    if duration < 8:
        # Too short to bother — keep the whole thing on the speaker.
        return []

    # Show the LLM a per-second timeline grid so it can think in time.
    grid = "\n".join(
        f"[{segment.start + i:.2f}-{min(segment.start + i + 1, segment.end):.2f}] ..."
        for i in range(int(duration))
    )

    resp = await client.chat.completions.create(
        model=settings.openai_reasoning_model,
        response_format={"type": "json_object"},
        temperature=0.3,
        messages=[
            {"role": "system", "content": SCENE_SYSTEM},
            {"role": "user", "content": SCENE_USER.format(
                seg_start=segment.start,
                seg_end=segment.end,
                lines=f"FULL TRANSCRIPT: \"{segment.transcript}\"\n\nTIME GRID:\n{grid}",
            )},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")

    raw = data.get("windows", [])
    out: list[InsertionWindow] = []
    for w in raw:
        try:
            win = InsertionWindow(
                start=float(w["start"]),
                end=float(w["end"]),
                rationale=str(w.get("rationale", "")),
                transcript=str(w.get("transcript", "")),
                priority=float(w.get("priority", 0.5)),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("dropping malformed insertion window {}: {}", w, e)
            continue

        # Enforce hard constraints regardless of what the LLM returned.
        win.start = max(win.start, segment.start + 2.5)
        win.end = min(win.end, segment.end - 1.5)
        dur = win.end - win.start
        if dur < 2.5:
            continue
        if dur > 5.0:
            win.end = win.start + 5.0
        out.append(win)

    # Deduplicate overlaps, keep highest priority.
    out.sort(key=lambda w: w.start)
    deduped: list[InsertionWindow] = []
    for w in out:
        if deduped and w.start < deduped[-1].end + 1.0:
            if w.priority > deduped[-1].priority:
                deduped[-1] = w
            continue
        deduped.append(w)

    # Trim to budget.
    deduped.sort(key=lambda w: w.priority, reverse=True)
    return deduped[:max_windows]
