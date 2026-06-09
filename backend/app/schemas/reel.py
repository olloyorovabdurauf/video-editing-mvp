"""Pydantic schemas for the reel pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class AspectRatio(str, Enum):
    VERTICAL = "9:16"      # Reels / Shorts / TikTok
    HORIZONTAL = "16:9"    # YouTube long-form
    SQUARE = "1:1"


class Mood(str, Enum):
    ENERGETIC = "energetic"
    CALM = "calm"
    INSPIRATIONAL = "inspirational"
    DRAMATIC = "dramatic"
    NEUTRAL = "neutral"


CaptionStyle = Literal["karaoke", "popup", "minimal", "none"]


class ReelCreateRequest(BaseModel):
    """Input the client posts to /api/v1/reels."""

    source_url: HttpUrl = Field(..., description="YouTube link or any direct video URL")
    aspect: AspectRatio = AspectRatio.VERTICAL
    target_count: int = Field(3, ge=1, le=10, description="How many reels to extract")
    max_duration_s: int = Field(60, ge=10, le=180)

    # Quality knobs — sensible defaults that produce a viral-looking reel.
    caption_style: CaptionStyle = "karaoke"
    smart_crop: bool = Field(True, description="Face-tracking pan-and-scan vs naive center")
    add_music: bool = True

    # Stock b-roll (Pexels) — free, fast, the default.
    add_broll: bool = True

    # AI b-roll — premium tier. Off by default. When enabled the user must
    # have credits + a per-job budget. Falls back to stock on any failure.
    use_ai_broll: bool = Field(
        False,
        description="Premium: cinematic AI-generated b-roll. Costs credits."
    )
    ai_broll_budget_usd: float = Field(
        4.0, ge=0, le=20,
        description="Hard cap on AI b-roll spend for this job, USD"
    )

    mood: Mood | None = Field(None, description="Override auto-detected mood")
    prompt: str | None = Field(
        None,
        description="Optional natural-language directive, e.g. "
                    "'focus on funny moments, keep clips under 30s'",
    )

    # Identity — for credit accounting. In dev we accept anonymous; prod
    # should require an authenticated user via the auth dep.
    user_id: str = Field("anonymous", description="Owner; debited for credit usage")


class Segment(BaseModel):
    start: float
    end: float
    hook_score: float = Field(..., ge=0, le=1)
    reason: str
    transcript: str


class BRollClip(BaseModel):
    segment_index: int
    start_offset: float        # seconds into the segment
    duration: float
    keyword: str
    source_url: HttpUrl
    provider: Literal["pexels", "pixabay"]


class ReelArtifact(BaseModel):
    segment: Segment
    output_url: HttpUrl | str
    thumbnail_url: HttpUrl | str | None = None
    broll: list[BRollClip] = []


class ReelJobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    GENERATING_BROLL = "generating_broll"
    RENDERING = "rendering"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ReelJobResponse(BaseModel):
    job_id: str
    status: ReelJobStatus
    progress: float = Field(0.0, ge=0, le=1)
    message: str | None = None
    artifacts: list[ReelArtifact] = []
