"""Pydantic schemas for the reel pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


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
    """Input the client posts to /api/v1/reels.

    Exactly one source must be given:
      - source_url: a YouTube link or direct video URL (we fetch it), or
      - upload_key: the R2 key returned by POST /uploads (user-uploaded file).
    """

    source_url: HttpUrl | None = Field(
        None, description="YouTube link or any direct video URL")
    upload_key: str | None = Field(
        None, max_length=512, description="R2 key from POST /uploads for an uploaded file")
    aspect: AspectRatio = AspectRatio.VERTICAL
    target_count: int = Field(3, ge=1, le=10, description="How many reels to extract")
    # Clips are COMPLETE short-form pieces (full hook→context→value→payoff arc),
    # not highlight fragments. Default 45-60s so each works as a standalone Reel.
    max_duration_s: int = Field(60, ge=15, le=180)
    min_duration_s: int = Field(45, ge=15, le=120,
                                description="Floor so clips are complete, not 5-10s hooks")

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

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ReelCreateRequest":
        if bool(self.source_url) == bool(self.upload_key):
            raise ValueError("provide exactly one of source_url or upload_key")
        return self

    @model_validator(mode="after")
    def _sane_durations(self) -> "ReelCreateRequest":
        if self.min_duration_s > self.max_duration_s:
            raise ValueError("min_duration_s must be <= max_duration_s")
        return self

    @property
    def is_upload(self) -> bool:
        return self.upload_key is not None


class Segment(BaseModel):
    start: float
    end: float
    # Per-clip scores (0..1). A clip is selected on a complete-story basis, not
    # on hook alone: completeness + value matter as much as the hook.
    hook_score: float = Field(0.0, ge=0, le=1)
    value_score: float = Field(0.0, ge=0, le=1)
    completeness_score: float = Field(0.0, ge=0, le=1)
    payoff_score: float = Field(0.0, ge=0, le=1)
    score: float = Field(0.0, ge=0, le=1)          # overall, completeness-weighted
    reason: str = ""                               # why this works as a standalone clip
    summary: str = ""                              # one line: the complete idea it covers
    transcript: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


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
    # AI-generated, ready-to-post metadata for this clip (best-effort; empty if
    # generation fails). Produced in the source video's language.
    title: str = ""
    caption: str = ""
    hashtags: list[str] = []


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
    model_config = ConfigDict(extra="ignore")   # tolerate internal job-state keys

    job_id: str
    status: ReelJobStatus
    progress: float = Field(0.0, ge=0, le=1)
    message: str | None = None
    # Streaming render: clips appear in `artifacts` as each finishes. These let
    # the UI show "X of Y clips ready" while the rest are still rendering.
    total_clips: int | None = None
    completed_clips: int | None = None
    artifacts: list[ReelArtifact] = []
    # Observability (populated on completion)
    cost_usd: float | None = None
    processing_time_s: float | None = None
    audio_minutes: float | None = None
