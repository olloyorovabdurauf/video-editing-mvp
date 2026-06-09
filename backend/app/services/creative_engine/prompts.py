"""
Prompt engineering for video generation.

The structure here is the single biggest lever on b-roll quality. Models
hallucinate less and produce more cinematic output when prompts follow a
consistent skeleton:

    [SHOT TYPE], [SUBJECT + ACTION], [SETTING], [LIGHTING],
    [CAMERA MOTION], [STYLE MODIFIERS], [TECHNICAL]

Naive prompt → fast cuts, generic noise, plasticky textures.
Structured prompt → coherent shots that cut into the timeline.

We use GPT-4o as a *prompt compiler*: it reads the segment transcript and
emits a strict-JSON VisualPrompt object, which we then render into the
provider-specific prompt string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings


SHOT_TYPES = Literal[
    "cinematic drone shot", "slow dolly-in", "close-up", "macro shot",
    "wide establishing shot", "handheld walking shot", "static tripod shot",
    "overhead top-down shot", "tracking shot", "first-person POV",
]

LIGHTING = Literal[
    "golden hour", "blue hour", "soft natural light", "neon-lit night",
    "harsh midday sun", "moody low-key", "high-key studio", "candle-lit",
    "volumetric god rays", "overcast diffuse",
]

STYLES = Literal[
    "hyper-realistic cinematic", "documentary 35mm film", "music-video grade",
    "minimalist editorial", "futuristic high-tech", "warm nostalgic super-8",
    "moody noir", "bright pastel commercial",
]


@dataclass
class VisualPrompt:
    """Structured representation; rendered to a string per provider."""

    shot: str               # one of SHOT_TYPES
    subject: str            # "a young engineer typing at a backlit laptop"
    setting: str            # "in a minimalist concrete loft at night"
    lighting: str           # one of LIGHTING
    motion: str             # "slow push-in" | "static" | "subtle handheld sway"
    style: str              # one of STYLES
    technical: str = "shallow depth of field, 24fps, color graded"
    negative: str = "text, logos, watermark, lowres, deformed hands, jittery motion"

    def render(self, provider: str = "default") -> str:
        """Provider-specific phrasing. Runway likes commas, Pika likes clauses."""
        core = (
            f"{self.shot}, {self.subject}, {self.setting}, "
            f"{self.lighting} lighting, {self.motion}, "
            f"{self.style} style, {self.technical}"
        )
        if provider == "runway":
            return core
        if provider == "higgsfield":
            return core
        return core


COMPILER_SYSTEM = """\
You translate a moment of dialogue into a single, professionally composed
b-roll shot. You DO NOT illustrate the words literally — you find a
*complementary* visual that elevates the meaning.

Hard rules:
- Subject must be filmable. No abstract concepts ("freedom"); show what
  freedom *looks like* ("a single hiker cresting a ridge at dawn").
- Match emotional tone to lighting (calm → soft natural; tense → low-key).
- No people speaking to camera — that's what the source video already shows.
- Avoid logos, recognizable brands, recognizable faces.
- Return STRICT JSON, no prose, no markdown fences.
"""

COMPILER_USER = """\
The speaker says (within a {duration:.0f}-second window):

\"\"\"{transcript}\"\"\"

Higher-level context for the whole reel: {reel_context}

Emit JSON with this exact shape:
{{
  "shot":      "<one of: cinematic drone shot | slow dolly-in | close-up | macro shot | wide establishing shot | handheld walking shot | static tripod shot | overhead top-down shot | tracking shot | first-person POV>",
  "subject":   "<concrete filmable subject + their action, 8-16 words>",
  "setting":   "<where this takes place, 4-10 words, starting with 'in' or 'on'>",
  "lighting":  "<one of: golden hour | blue hour | soft natural light | neon-lit night | harsh midday sun | moody low-key | high-key studio | candle-lit | volumetric god rays | overcast diffuse>",
  "motion":    "<camera motion, 2-6 words>",
  "style":     "<one of: hyper-realistic cinematic | documentary 35mm film | music-video grade | minimalist editorial | futuristic high-tech | warm nostalgic super-8 | moody noir | bright pastel commercial>"
}}
"""


async def compile_visual_prompt(
    transcript: str,
    *,
    duration: float,
    reel_context: str = "",
) -> VisualPrompt:
    """LLM-driven: dialogue line → structured VisualPrompt."""
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    resp = await client.chat.completions.create(
        model=settings.openai_reasoning_model,
        response_format={"type": "json_object"},
        temperature=0.7,                    # creative, but not random
        messages=[
            {"role": "system", "content": COMPILER_SYSTEM},
            {"role": "user",   "content": COMPILER_USER.format(
                duration=duration,
                transcript=transcript.strip(),
                reel_context=reel_context or "(general)",
            )},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    try:
        return VisualPrompt(
            shot=data["shot"], subject=data["subject"], setting=data["setting"],
            lighting=data["lighting"], motion=data["motion"], style=data["style"],
        )
    except KeyError as e:
        logger.warning("prompt compile missing key {}; falling back to generic", e)
        return VisualPrompt(
            shot="slow dolly-in",
            subject="abstract motion graphics complementing the theme",
            setting="in a softly-lit minimal environment",
            lighting="soft natural light",
            motion="slow push-in",
            style="hyper-realistic cinematic",
        )
