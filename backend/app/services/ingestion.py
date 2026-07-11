"""
Source ingestion — the ONE place that knows how to fetch a video.

Why a dedicated module
----------------------
yt-dlp is the riskiest dependency we have: it tracks an adversarial, moving
target (YouTube) and breaks without warning. Isolating it here means:
  - the pipeline only ever calls `download_source(url, dir)` — implementation
    can change (add a proxy, swap libraries) without touching task code,
  - failures map to *clear, user-facing* messages instead of leaking tracebacks,
  - we can monitor the yt-dlp version and warn when it goes stale.

Two paths:
  - Direct media URL (…/clip.mp4) → plain HTTP with header-strategy fallback.
  - Platform URL (YouTube/Vimeo/…) → yt-dlp (hardened: timeouts, retries,
    optional residential proxy).
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

from app.config import get_settings

_DIRECT_MEDIA_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".m4a", ".mp3", ".wav")
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADER_STRATEGIES: tuple[dict[str, str], ...] = (
    {   # 1. Fully-formed browser fingerprint (passes most WAFs)
        "User-Agent": _BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,video/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "video", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Site": "cross-site",
    },
    {"User-Agent": "curl/8.7.1", "Accept": "*/*"},                  # 2. curl-like
    {"User-Agent": "ReelForge/1.0 (+https://reelforge.app)"},       # 3. honest bot
)
_BLOCKED_STATUSES = frozenset({401, 403, 429})

# yt-dlp must stay current or YouTube extraction breaks. Warn if the pinned
# build is older than this many days (a nudge to bump the pin / Dependabot).
_YTDLP_STALE_DAYS = 45


class IngestionError(Exception):
    """A download failure with a message safe to show the end user."""

    def __init__(self, user_message: str, *, retryable: bool = False, cause: Exception | None = None):
        self.user_message = user_message
        self.retryable = retryable
        super().__init__(f"{user_message}" + (f" (cause: {cause})" if cause else ""))


# ---------------------------------------------------------------------------
# Version monitoring
# ---------------------------------------------------------------------------

def ytdlp_version() -> str:
    import yt_dlp
    return getattr(yt_dlp.version, "__version__", "unknown")


def warn_if_ytdlp_stale() -> None:
    """yt-dlp versions are date-stamped (YYYY.MM.DD). Log loudly if stale."""
    v = ytdlp_version()
    m = re.match(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", v)
    if not m:
        return
    try:
        built = _dt.date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return
    age = (_dt.date.today() - built).days
    if age > _YTDLP_STALE_DAYS:
        logger.warning(
            "yt-dlp {} is {} days old — YouTube extraction may break. Bump the "
            "pin in requirements.txt (Dependabot recommended).", v, age,
        )


# ---------------------------------------------------------------------------
# Routing + download
# ---------------------------------------------------------------------------

def looks_like_direct_media(url: str) -> bool:
    return urlparse(url).path.lower().endswith(_DIRECT_MEDIA_EXTS)


def fetch_upload(upload_key: str, work_dir: Path) -> Path:
    """
    Pull a user-uploaded file from object storage (R2) into work_dir so it
    enters the *exact same* pipeline as a URL/YouTube source. No yt-dlp, no
    SSRF surface — it's our own private bucket.
    """
    from app.services.storage import get_storage

    work_dir.mkdir(parents=True, exist_ok=True)
    dst = work_dir / "source.mp4"
    try:
        get_storage().download_to(upload_key, dst)
    except Exception as e:
        raise IngestionError("We couldn't read your uploaded file. Please re-upload and try again.",
                             cause=e) from e
    if not dst.exists() or dst.stat().st_size == 0:
        raise IngestionError("The uploaded file is empty or missing.")
    return dst


def download_source(url: str, work_dir: Path) -> Path:
    """
    Fetch `url` into `work_dir`, returning the local file path. Raises
    IngestionError (with a user-safe message) on failure.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        if looks_like_direct_media(url):
            src = _download_direct(url, work_dir / "source.mp4")
        else:
            warn_if_ytdlp_stale()
            src = _download_via_ytdlp(url, str(work_dir / "source.%(ext)s"))
        if not src.exists() or src.stat().st_size == 0:
            raise IngestionError("The download produced no file. Please check the URL.")
        return src
    except IngestionError:
        raise
    except Exception as e:
        raise _to_user_error(e) from e


def _download_direct(url: str, dst: Path) -> Path:
    import httpx

    dst.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=15.0)
    last_status: int | None = None
    for headers in _HEADER_STRATEGIES:
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout, headers=headers) as resp:
                if resp.status_code in _BLOCKED_STATUSES:
                    last_status = resp.status_code
                    continue
                resp.raise_for_status()
                with dst.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1 << 16):
                        f.write(chunk)
                return dst
        except httpx.HTTPStatusError as e:
            last_status = e.response.status_code
            if e.response.status_code in _BLOCKED_STATUSES:
                continue
            raise
    raise IngestionError(
        "Couldn't download from this host — it's blocking automated requests. "
        "Try a different source or upload the file directly.",
        retryable=False,
    )


def _base_ydl_opts(out_tmpl: str, *, proxy: str | None) -> dict:
    opts = {
        "outtmpl": out_tmpl,
        "quiet": True, "noprogress": True,
        "socket_timeout": 30,
        "retries": 5, "extractor_retries": 3, "fragment_retries": 10,
        "concurrent_fragment_downloads": 4,
    }
    if proxy:
        opts["proxy"] = proxy
    return opts


def _download_via_ytdlp(url: str, out_tmpl: str) -> Path:
    import yt_dlp

    ydl_opts = {
        **_base_ydl_opts(out_tmpl, proxy=get_settings().ytdlp_proxy),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)).with_suffix(".mp4")


# ---------------------------------------------------------------------------
# Fast path for platform videos behind the (slow, GB-metered) residential
# proxy: pull ONLY the audio for transcription/analysis, then ONLY the video
# sections the AI actually selected. A 60-min video drops from ~800MB to
# ~25MB (audio) + ~20MB per clip — and the sections download in PARALLEL
# across distinct proxy sessions, so line speeds aggregate.
# ---------------------------------------------------------------------------

def proxy_for_session(k: int) -> str | None:
    """
    Derive a distinct sticky-session proxy URL from the configured one.
    Webshare-style usernames end in "-<session>" (e.g. user-gb-1); each session
    number is a different residential exit line. Rewriting the suffix lets
    parallel downloads use separate lines instead of queueing on one.
    Falls back to the base proxy when the format doesn't match.
    """
    base = get_settings().ytdlp_proxy
    if not base or k <= 0:
        return base
    m = re.match(r"^(https?://)([^:@/]+):([^@/]+)@(.+)$", base)
    if not m:
        return base
    scheme, user, pw, host = m.groups()
    m2 = re.match(r"^(.*-)(\d+)$", user)
    if not m2:
        return base
    return f"{scheme}{m2.group(1)}{int(m2.group(2)) + k}:{pw}@{host}"


def _audio_opts(out_tmpl: str, *, proxy: str | None) -> dict:
    # ~48-96kbps voice audio is all Whisper needs; ~25MB/hour vs 800MB/hour.
    return {
        **_base_ydl_opts(out_tmpl, proxy=proxy),
        "format": "ba[abr<=96]/wa[abr>=48]/ba/b",
    }


def _section_opts(out_tmpl: str, start: float, end: float, *, proxy: str | None) -> dict:
    from yt_dlp.utils import download_range_func

    return {
        **_base_ydl_opts(out_tmpl, proxy=proxy),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "download_ranges": download_range_func(None, [(start, end)]),
        # Re-encode the section to EXACTLY [start, end] — the output then IS
        # the frame-accurate cut, so the render pipeline skips its cut pass.
        "force_keyframes_at_cuts": True,
    }


def download_audio(url: str, work_dir: Path) -> Path:
    """Audio-only download for transcription (platform URLs)."""
    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        with yt_dlp.YoutubeDL(_audio_opts(str(work_dir / "audio.%(ext)s"),
                                          proxy=get_settings().ytdlp_proxy)) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except IngestionError:
        raise
    except Exception as e:
        raise _to_user_error(e) from e
    if not path.exists() or path.stat().st_size == 0:
        raise IngestionError("The download produced no audio. Please check the URL.")
    return path


def download_section(url: str, work_dir: Path, index: int,
                     start: float, end: float, *, session: int = 0) -> Path:
    """
    Download one selected window of the video, frame-accurate, through its own
    proxy session. Returns the ready-to-use clip file (already cut to [start, end]).
    """
    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    tmpl = str(work_dir / f"section_{index}.%(ext)s")
    try:
        with yt_dlp.YoutubeDL(_section_opts(tmpl, start, end,
                                            proxy=proxy_for_session(session))) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    except IngestionError:
        raise
    except Exception as e:
        raise _to_user_error(e) from e
    if not path.exists():
        # yt-dlp may decorate section filenames — recover by prefix.
        candidates = sorted(work_dir.glob(f"section_{index}*.mp4"))
        path = candidates[0] if candidates else path
    if not path.exists() or path.stat().st_size == 0:
        raise IngestionError("The section download produced no file.")
    return path


def _to_user_error(e: Exception) -> IngestionError:
    """Map a raw yt-dlp/HTTP error to a clear, user-facing message."""
    msg = str(e).lower()
    # BEFORE the age/sign-in branch: YouTube's bot challenge reads "Sign in to
    # confirm you're not a bot" — it's an IP-reputation check, not age gating,
    # and matching on "sign in" first mislabeled it for users.
    if "not a bot" in msg:
        return IngestionError(
            "YouTube temporarily challenged our downloader on this video. "
            "Try again in a minute — if it keeps happening, upload the file directly.",
            retryable=True, cause=e,
        )
    if "402" in msg or "payment required" in msg:
        # The residential proxy plan is out of bandwidth/payment (Webshare 402).
        # Retrying can't help until the plan is topped up — fail honestly.
        return IngestionError(
            "Our video download service is temporarily out of capacity. "
            "We're restoring it — please try again soon, or upload the file directly.",
            retryable=False, cause=e,
        )
    if "private" in msg or "unavailable" in msg or "removed" in msg or "does not exist" in msg:
        return IngestionError("This video is private, removed, or region-locked.", cause=e)
    # NOTE: match precise phrases — a bare "age" substring matched "API pAGE"
    # and mislabeled proxy failures as age-restriction.
    if "sign in" in msg or "age-restricted" in msg or "confirm your age" in msg or "login required" in msg:
        return IngestionError("This video requires sign-in (age-restricted) and can't be processed.", cause=e)
    if "403" in msg or "forbidden" in msg or "bot" in msg or "429" in msg or "too many requests" in msg:
        return IngestionError(
            "The source is temporarily rate-limiting downloads. Try again in a few "
            "minutes, or upload the file directly.",
            retryable=True, cause=e,
        )
    if "timed out" in msg or "timeout" in msg:
        return IngestionError("The download timed out. The source may be slow — try again.", retryable=True, cause=e)
    return IngestionError("Couldn't download this video. Make sure the URL is public and correct.", cause=e)
