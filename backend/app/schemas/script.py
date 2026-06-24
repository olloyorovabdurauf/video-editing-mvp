"""
Schemas for AI Reels *script* generation.

This is a text product, separate from the video-clip pipeline: given a topic +
language + style, we generate a full, retention-engineered Reels script (not a
clip cut from existing footage). Every script follows a fixed 4-part structure
(hook → problem → value → payoff+CTA) sized for 40-60s so the idea is fully
developed and the hook's promise is always paid off.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class ScriptLanguage(str, Enum):
    EN = "en"
    RU = "ru"
    UZ = "uz"


class ScriptStyle(str, Enum):
    # Founder / build-in-public: journey, AI future, startup lessons, product building.
    FOUNDER = "founder_building_in_public"
    # Educational: problem → insight → framework → example.
    EDUCATIONAL = "educational"


# The mandatory retention skeleton. Order + presence are validated.
SectionName = Literal["hook", "problem", "value", "payoff"]
REQUIRED_SECTIONS: tuple[SectionName, ...] = ("hook", "problem", "value", "payoff")


class ScriptGenerateRequest(BaseModel):
    """Input to POST /api/v1/scripts."""

    topic: str | None = Field(
        default=None, max_length=400,
        description="What the reel is about. If omitted, the model proposes a "
                    "strong on-brand topic for the niche/style.",
    )
    language: ScriptLanguage = ScriptLanguage.UZ
    style: ScriptStyle = ScriptStyle.FOUNDER
    duration_seconds: int = Field(
        default=60, ge=40, le=75,
        description="Target length. Hard floor 40s so the idea is fully developed "
                    "— the fix for thin 15s reels with no payoff.",
    )
    platform: str = Field(default="instagram_reels")
    niche: str = Field(
        default="AI video editing / short-form content startup",
        max_length=200,
        description="Context used for relevance and for inventing a topic when none is given.",
    )
    # Clients may send a user_id; the auth token overrides it in the endpoint.
    user_id: str | None = None


class ScriptSection(BaseModel):
    """One beat of the script, time-boxed, with the spoken words + visual direction."""

    name: SectionName
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    voiceover: str = Field(description="Exact words spoken, in the target language.")
    visual: str = Field(description="On-screen text / b-roll / action direction.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def time_range(self) -> str:
        return f"{int(round(self.start_s))}-{int(round(self.end_s))}s"


class Caption(BaseModel):
    """The Instagram caption: strong first line, short body, explicit CTA."""

    hook: str
    body: str
    cta: str

    def render(self, hashtags: list[str]) -> str:
        tags = " ".join(h if h.startswith("#") else f"#{h}" for h in hashtags)
        parts = [self.hook.strip(), self.body.strip(), self.cta.strip()]
        if tags:
            parts.append(tags)
        return "\n\n".join(p for p in parts if p)


class ScriptResponse(BaseModel):
    """A complete, ready-to-shoot Reels script."""

    title: str
    hook: str                       # the spoken hook line (target language)
    script: str                     # full read-through voiceover (target language)
    sections: list[ScriptSection]
    caption: Caption
    hashtags: list[str] = Field(default_factory=list)
    language: ScriptLanguage
    style: ScriptStyle
    duration_seconds: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted(self) -> str:
        """Human-readable block in the exact requested output format."""
        breakdown = "\n".join(
            f"[{s.time_range}] {s.name.upper()}\n"
            f"  VO: {s.voiceover}\n"
            f"  ON-SCREEN: {s.visual}"
            for s in self.sections
        )
        return (
            f"TITLE:\n{self.title}\n\n"
            f"HOOK:\n{self.hook}\n\n"
            f"SCRIPT:\n{self.script}\n\n"
            f"SCENE BREAKDOWN:\n{breakdown}\n\n"
            f"CAPTION:\n{self.caption.render(self.hashtags)}\n\n"
            f"LANGUAGE:\n{self.language.value}"
        )
