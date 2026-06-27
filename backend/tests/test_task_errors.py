"""Error classification in the pipeline (OpenAI quota → fail fast, friendly msg)."""
from __future__ import annotations

from app.tasks import video_tasks as vt


def test_quota_error_detected():
    assert vt._is_quota_error("Error code: 429 - {'code': 'insufficient_quota'}")
    assert vt._is_quota_error("You exceeded your current quota, please check your plan")
    assert not vt._is_quota_error("Error code: 500 - internal server error")
    assert not vt._is_quota_error("connection timed out")


def test_quota_message_is_friendly():
    assert "provider quota" in vt._AI_QUOTA_MSG.lower()
    assert "insufficient_quota" not in vt._AI_QUOTA_MSG   # no raw provider jargon
