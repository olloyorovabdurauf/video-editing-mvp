"""
Music selector.

Layered design — substitutable from cheapest to most sophisticated:

  1. Local curated library (free, instant) — what ships in MVP.
  2. Per-job music search via a stock provider (Epidemic, Artlist) — v2.
  3. Generative score (Stable Audio, MusicGen) — v3, only when user pays for it.

Mood detection itself stays close to the LLM that already analyzed the
transcript: we ask GPT-4o to label segments with one of the five Moods
defined in schemas/reel.py, and the selector maps mood → tracks.

Library layout
--------------
    storage/music/
        energetic/      *.mp3
        calm/           *.mp3
        inspirational/  *.mp3
        dramatic/       *.mp3
        neutral/        *.mp3
        _licenses.json  # per-track license metadata

License metadata is REQUIRED for any production reel. We do not commit to
making a music selection if license metadata is missing — silence is safer
than a copyright claim.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.reel import Mood, Segment


@dataclass(frozen=True)
class Track:
    path: Path
    mood: Mood
    title: str
    artist: str
    license: str               # "CC0", "royalty_free", "licensed:{id}"
    duration_s: float
    bpm: int | None = None


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _library_root() -> Path:
    return get_settings().storage_local_dir / "music"


def _load_license_meta() -> dict[str, dict]:
    meta_path = _library_root() / "_licenses.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error("music license metadata corrupt: {}", e)
        return {}


def list_tracks(mood: Mood) -> list[Track]:
    """List licensed tracks for a mood. Unlicensed files are silently skipped."""
    root = _library_root() / mood.value
    if not root.exists():
        return []
    meta = _load_license_meta()
    tracks: list[Track] = []
    for p in sorted(root.glob("*.mp3")):
        m = meta.get(p.name)
        if not m or "license" not in m:
            logger.warning("skipping unlicensed track {}", p.name)
            continue
        tracks.append(Track(
            path=p,
            mood=mood,
            title=m.get("title", p.stem),
            artist=m.get("artist", "unknown"),
            license=m["license"],
            duration_s=float(m.get("duration_s", 0.0)),
            bpm=m.get("bpm"),
        ))
    return tracks


# ---------------------------------------------------------------------------
# Mood inference (segment → Mood)
# ---------------------------------------------------------------------------

MOOD_PROMPT = """\
Classify the mood of the following short video segment for music selection.
Return JSON: {"mood": "energetic|calm|inspirational|dramatic|neutral"}.

Pick one:
- energetic: fast pace, excitement, urgency, comedy, action
- calm: meditative, reflective, soothing
- inspirational: aspirational, motivational, uplifting build
- dramatic: tense, weighty, serious moment
- neutral: informational, even-keeled, news-like

Transcript:
\"\"\"{transcript}\"\"\"
"""


async def infer_mood(segment: Segment) -> Mood:
    """LLM-driven mood classification. Costs ~$0.001/call."""
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model=settings.openai_reasoning_model,
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[{"role": "user", "content": MOOD_PROMPT.format(transcript=segment.transcript)}],
    )
    try:
        raw = json.loads(resp.choices[0].message.content or "{}")
        return Mood(raw.get("mood", "neutral"))
    except (ValueError, json.JSONDecodeError):
        return Mood.NEUTRAL


# ---------------------------------------------------------------------------
# Public selector
# ---------------------------------------------------------------------------

async def pick_track(
    segment: Segment,
    *,
    override_mood: Mood | None = None,
    seed: int | None = None,
) -> Track | None:
    """
    Choose a music bed for a segment. Returns None if no licensed track is
    available — the renderer treats None as "ship without music", not an error.
    """
    mood = override_mood or await infer_mood(segment)
    tracks = list_tracks(mood)
    if not tracks:
        # Soft-degrade: try neutral.
        if mood != Mood.NEUTRAL:
            tracks = list_tracks(Mood.NEUTRAL)
        if not tracks:
            logger.info("no music available for mood {} — shipping silent", mood.value)
            return None

    rng = random.Random(seed) if seed is not None else random
    chosen = rng.choice(tracks)
    logger.info("music: mood={} → {!r} by {}", mood.value, chosen.title, chosen.artist)
    return chosen
