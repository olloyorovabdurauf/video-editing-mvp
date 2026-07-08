# Multi-Agent Clip Pipeline (v5)

## Current architecture & why it fails
`segment_picker.pick_segments` = ONE LLM call doing story-map + selection +
scoring on a COMPRESSED (~4s bucket) transcript, then `_finalize_window` snaps
boundaries to sentences. Failure mode: nothing ever re-reads the FINAL window's
exact words. Selection quality is verified at pick time on compressed text, so
a clip that lost its setup/payoff during snapping — or was scored on a lossy
summary — renders anyway.

## New architecture (staged, each stage a separate LLM call)
1. **Analyst+Segmenter** `_story_graph(transcript)` → complete stories with
   {topic, beginning/context/explanation/evidence/lesson/conclusion presence,
   start, end}. Reads the whole (compressed) transcript. Never selects.
2. **Selector** `_select_from_stories(graph, n)` → candidates scored on 10
   dims (hook, curiosity, story, meaning, emotion, value, retention,
   conclusion, standalone, virality) + overall. Only from complete stories.
3. **Boundary finalize** (existing `_finalize_window` + completion grace).
4. **Quality Reviewer** `_review_clips(finalized)` → gets the EXACT word-level
   text of each finalized window; per clip returns
   {verdict: approve|reject, start_nudge, end_nudge, reason} judging: would a
   cold Instagram viewer get what/why/complete explanation/conclusion?
   Nudges re-finalized; rejects replaced from unused stories (1 replacement
   round), else dropped. NEVER render an unapproved clip (if all rejected,
   fall back to top-scored with warning — never 0 clips for a video with speech).
5. **Editor** = existing render (smart crop/captions/music) — unchanged, only
   approved clips reach it.

Cache key: `segpick_v5`. Reviewer uses cheap model, batched single call.

## Subtitles (already conformant — keep)
ASR words are the source of truth (word-level karaoke ASS). LLM never
paraphrases. Sole exception stays: Whisper emits KAZAKH for Uzbek audio →
translation layer + script validation (Latin-uz enforced, regenerate-once)
until the user's Google STT key lands (native uz then skips translation).
language_guard covers auto-detect. Cropping: shipped (multi-face, margins,
EMA, anchor) — no change.

## Files
- `backend/app/services/segment_picker.py` — split into stages 1/2/4; keep
  `_finalize_window`, `_distribute_clips`, `_dedupe_overlap` as-is.
- `backend/tests/test_segment_picker_fallback.py` — stage tests (stub seams:
  story graph JSON → selector JSON → reviewer verdicts incl. reject+nudge).
- After ship: deploy ritual (scale + machine update --command), e2e bench.

## Status
- [ ] Implement stages in segment_picker.py
- [ ] Tests green (220 base)
- [ ] Deploy + ritual + livez
- [ ] E2E benchmark (boundaries + reviewer verdicts logged)
