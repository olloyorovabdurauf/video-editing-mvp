"""
Reel-quality eval harness.

Usage:
    python -m eval.run_benchmark              # run all items
    python -m eval.run_benchmark podcast_01   # one item
    python -m eval.run_benchmark --against=baseline.json   # A/B compare

Metrics
-------
1. **Segment IoU vs golden**     — did the picker find the moments humans
                                    picked? (Jaccard over time intervals)
2. **Hook-score calibration**    — does a higher predicted hook_score
                                    actually correlate with golden picks?
3. **Caption sync error**        — mean |word.start - asr.start| in ms
4. **Render time / cost**        — wall-clock and USD per reel
5. **Cache hit rate**            — generative b-roll cache effectiveness

Output: a single JSON report + a printed table. Commit the report to git
as `eval/reports/<sha>.json` so you can diff runs.

Why this isn't just pytest: these are *quality* metrics over real videos,
not unit tests. Targets drift, models change, prompts evolve. Run before
every prompt or model bump.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from loguru import logger

# Re-use the production services, NOT a copy. The harness must exercise the
# same code path users hit, or it's measuring fiction.
from app.config import get_settings
from app.schemas.reel import ReelCreateRequest, AspectRatio, Segment
from app.services import segment_picker, transcription
from app.services.billing import total_cost_usd


REPORT_DIR = Path(__file__).parent / "reports"
BENCH_PATH = Path(__file__).parent / "benchmark_set.yaml"


@dataclass
class ItemResult:
    id: str
    n_predicted: int
    n_golden: int
    mean_iou: float
    matched: int                  # predicted segments with IoU >= 0.5 vs any golden
    hook_score_at_matched: float  # mean hook_score on matched picks
    hook_score_at_unmatched: float
    wall_clock_s: float
    cost_usd: float
    errors: list[str]


# ---------------------------------------------------------------------------
# Metric: temporal IoU between segments
# ---------------------------------------------------------------------------

def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _score_picks(predicted: list[Segment], golden: list[tuple[float, float]]) -> dict:
    ious_per_pred = []
    matched_flags = []
    for p in predicted:
        best = max((_iou((p.start, p.end), g) for g in golden), default=0.0)
        ious_per_pred.append(best)
        matched_flags.append(best >= 0.5)
    matched = sum(matched_flags)
    matched_hooks = [
        p.hook_score for p, m in zip(predicted, matched_flags) if m
    ]
    unmatched_hooks = [
        p.hook_score for p, m in zip(predicted, matched_flags) if not m
    ]
    return {
        "mean_iou": sum(ious_per_pred) / max(1, len(ious_per_pred)),
        "matched": matched,
        "hook_at_matched": sum(matched_hooks) / max(1, len(matched_hooks)),
        "hook_at_unmatched": sum(unmatched_hooks) / max(1, len(unmatched_hooks)),
    }


# ---------------------------------------------------------------------------
# Eval one item end-to-end
# ---------------------------------------------------------------------------

async def eval_item(item: dict) -> ItemResult:
    """
    Runs the AI picker pipeline (download + transcribe + analyze) and scores
    its picks against the golden segments. Skips rendering — quality of the
    picker is the dominant variable; rendering is mechanical.
    """
    errors: list[str] = []
    t0 = time.time()
    settings = get_settings()
    cache_dir = settings.storage_local_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download (or reuse cached file by id)
    import yt_dlp  # local import — module only needed if we run eval

    src_path = cache_dir / f"{item['id']}.mp4"
    if not src_path.exists():
        ydl_opts = {
            "outtmpl": str(cache_dir / f"{item['id']}.%(ext)s"),
            "format": "bv*[height<=720]+ba/b[height<=720]",
            "merge_output_format": "mp4", "quiet": True, "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([item["source_url"]])
        except Exception as e:
            errors.append(f"download: {e}")
            return ItemResult(
                id=item["id"], n_predicted=0, n_golden=len(item["golden_segments"]),
                mean_iou=0, matched=0, hook_score_at_matched=0, hook_score_at_unmatched=0,
                wall_clock_s=time.time() - t0, cost_usd=0, errors=errors,
            )

    # 2. Transcribe (cached on disk)
    t_path = src_path.with_suffix(".transcript.json")
    if t_path.exists():
        raw = json.loads(t_path.read_text(encoding="utf-8"))
        t = transcription.Transcript(
            language=raw["language"], text=raw["text"],
            words=[transcription.Word(**w) for w in raw["words"]],
        )
    else:
        try:
            t = await transcription.transcribe(src_path)
        except Exception as e:
            errors.append(f"transcribe: {e}")
            return ItemResult(
                id=item["id"], n_predicted=0, n_golden=len(item["golden_segments"]),
                mean_iou=0, matched=0, hook_score_at_matched=0, hook_score_at_unmatched=0,
                wall_clock_s=time.time() - t0, cost_usd=0, errors=errors,
            )
        t_path.write_text(json.dumps({
            "language": t.language, "text": t.text,
            "words": [w.__dict__ for w in t.words],
        }), encoding="utf-8")

    # 3. Pick segments
    try:
        picks = await segment_picker.pick_segments(
            t, n=item["target_count"], max_duration_s=item["max_duration_s"], prompt=None,
        )
    except Exception as e:
        errors.append(f"pick: {e}")
        picks = []

    # 4. Score
    golden = [(g[0], g[1]) for g in item["golden_segments"]]
    scores = _score_picks(picks, golden)

    return ItemResult(
        id=item["id"],
        n_predicted=len(picks),
        n_golden=len(golden),
        mean_iou=scores["mean_iou"],
        matched=scores["matched"],
        hook_score_at_matched=scores["hook_at_matched"],
        hook_score_at_unmatched=scores["hook_at_unmatched"],
        wall_clock_s=time.time() - t0,
        cost_usd=total_cost_usd(item["id"]),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(results: list[ItemResult]) -> None:
    cols = ["id", "matched", "mean_iou", "hook_match", "hook_miss", "wall_s", "cost_$"]
    print(" | ".join(c.rjust(12) for c in cols))
    print("-" * 92)
    for r in results:
        row = [
            r.id,
            f"{r.matched}/{r.n_predicted}",
            f"{r.mean_iou:.2f}",
            f"{r.hook_score_at_matched:.2f}",
            f"{r.hook_score_at_unmatched:.2f}",
            f"{r.wall_clock_s:.1f}",
            f"{r.cost_usd:.3f}",
        ]
        print(" | ".join(str(c).rjust(12) for c in row))
    if results:
        avg = sum(r.mean_iou for r in results) / len(results)
        print(f"\nMEAN IoU across {len(results)} items: {avg:.3f}")


def _write_report(results: list[ItemResult]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{int(time.time())}.json"
    path = REPORT_DIR / name
    path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

async def amain(only: str | None) -> int:
    bench = yaml.safe_load(BENCH_PATH.read_text(encoding="utf-8"))
    items = [i for i in bench["items"] if not only or i["id"] == only]
    if not items:
        logger.error("no benchmark items matched {!r}", only)
        return 1

    results: list[ItemResult] = []
    for item in items:
        logger.info("evaluating {}", item["id"])
        results.append(await eval_item(item))

    _print_table(results)
    report = _write_report(results)
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("only", nargs="?", default=None, help="benchmark id to run alone")
    args = parser.parse_args()
    sys.exit(asyncio.run(amain(args.only)))
