"""
Schemas for AI Reels *script* generation.

A text product, separate from the video-clip pipeline: given a topic + language
+ content type + industry, we generate a complete, retention-engineered Reels
script (40-60s; 30s allowed for punchy formats) following a fixed arc
(hook → problem → value → payoff+CTA) so the idea is fully developed and the
hook's promise is always paid off.

The product is for creators, experts, educators, coaches, businesses and
marketers — NOT only founders. Voice/goal/tone adapt to the selected content
type; founder/build-in-public is just one option, used only when chosen.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class ScriptLanguage(str, Enum):
    EN = "en"
    RU = "ru"
    UZ = "uz"


class ContentType(str, Enum):
    """What kind of reel — drives voice, goal, audience framing and tone."""
    EDUCATIONAL = "educational"
    PERSONAL_BRAND = "personal_brand"
    FOUNDER_STORY = "founder_story"          # build-in-public — only when selected
    PRODUCT_MARKETING = "product_marketing"
    STORYTELLING = "storytelling"
    TUTORIAL = "tutorial"
    SALES = "sales"
    VIRAL_REEL = "viral_reel"


class Industry(str, Enum):
    GENERAL = "general"
    BUSINESS = "business"
    HEALTH = "health"
    EDUCATION = "education"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    OTHER = "other"


# The mandatory retention skeleton. Order + presence are validated.
SectionName = Literal["hook", "problem", "value", "payoff"]
REQUIRED_SECTIONS: tuple[SectionName, ...] = ("hook", "problem", "value", "payoff")


class ScriptGenerateRequest(BaseModel):
    """Input to POST /api/v1/scripts."""

    topic: str | None = Field(
        default=None, max_length=400,
        description="What the reel is about. If omitted, the model proposes a "
                    "strong, specific topic for the content type + industry.",
    )
    language: ScriptLanguage = ScriptLanguage.UZ
    content_type: ContentType = ContentType.EDUCATIONAL   # NOT founder by default
    industry: Industry = Industry.GENERAL
    duration_seconds: int = Field(
        default=60, ge=30, le=75,
        description="Target length (30/45/60). The script is sized to genuinely "
                    "fill it — the fix for thin reels with no payoff.",
    )
    # Optional overrides. When empty, the model infers them from content type +
    # industry + topic so every reel still matches a clear goal/audience/tone.
    audience: str | None = Field(default=None, max_length=200)
    goal: str | None = Field(default=None, max_length=200)
    tone: str | None = Field(default=None, max_length=120)
    platform: str = Field(default="instagram_reels")
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
    content_type: ContentType
    industry: Industry
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
