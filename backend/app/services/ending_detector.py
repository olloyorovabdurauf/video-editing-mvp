"""
Semantic Ending Detection — stop the clip at the FIRST strong natural ending.

Why this exists: boundary repair + the reviewer's finish gate guarantee a clip
never ends BEFORE the idea completes — but nothing detected that the payoff had
already landed. Clips kept running into the next example, a repetition, or a
topic change, which kills retention. A professional editor cuts the moment the
audience has received the payoff; everything after belongs to the NEXT clip.

Contract of this module:
- It only ever TRIMS (or keeps) a clip end — extension is boundary repair's job.
- It never trims below the hard fragment floor (0.6x the minimum duration).
- The cut lands exactly on the chosen sentence's final word timestamp.
- Look-ahead: the model also sees what comes AFTER the current end, so
  "the next 15s is filler/another story" is evidence FOR stopping, and genuine
  continuation is visible too. Low confidence -> keep the reviewed boundary.
"""
from __future__ import annotations

import json

from loguru import logger

from app.config import get_settings
from app.schemas.reel import Segment
from app.services.transcription import Transcript

_CONFIDENCE_THRESHOLD = 0.6
_LOOK_AHEAD_S = 20.0
_MIN_TRIM_S = 1.5            # ignore sub-1.5s trims (noise)

_ENDING_PROMPT = """\
You are a retention editor. For each clip you get its sentences NUMBERED with
end-times, then the content that comes AFTER the current cut (marked AFTER-END,
not in the clip). Find the FIRST STRONG NATURAL ENDING: the sentence that
delivers the final conclusion, the main lesson, the payoff, the answer to the
hook, a strong memorable quote, or the emotional landing.
The clip must stop THERE — never continue into: a new example, a topic change,
repetition, another story, a different question, thanks, or filler. Those
belong to the next clip. If the AFTER-END content shows the idea genuinely
continues, or every sentence adds value to the very end, keep the current cut
(use the LAST sentence index with low-to-medium confidence).
Return STRICT JSON:
{"endings": [{"i": <clip number>, "end_sentence": <sentence number of the
strong ending>, "confidence": <0..1 that stopping there is BETTER than the
current cut>, "reason": "<short>"}]}"""


def _sentences_in(transcript: Transcript, start: float, end: float) -> list[tuple[float, str]]:
    """Clip text split into sentences via punctuation + pauses → [(end_time, text)]."""
    from app.services.segment_picker import _PAUSE_BOUNDARY_S, _is_sentence_end

    words = [w for w in transcript.words if start <= w.start < end]
    out: list[tuple[float, str]] = []
    buf: list[str] = []
    for i, w in enumerate(words):
        buf.append(w.text)
        gap_after = (words[i + 1].start - w.end) if i + 1 < len(words) else 0.0
        if _is_sentence_end(w.text) or gap_after >= _PAUSE_BOUNDARY_S or i == len(words) - 1:
            out.append((w.end, " ".join(buf)))
            buf = []
    return out


async def refine_endings(client, transcript: Transcript, segs: list[Segment],
                         *, min_s: float) -> list[Segment]:
    """Trim each clip to its first strong natural ending (batched, one call)."""
    if not segs:
        return list(segs)
    settings = get_settings()
    hard_floor = max(15.0, 0.6 * min_s)

    per_clip_sentences: list[list[tuple[float, str]]] = []
    blocks: list[str] = []
    for ci, s in enumerate(segs):
        sents = _sentences_in(transcript, s.start, s.end)
        per_clip_sentences.append(sents)
        lines = [f"  S{si} [ends {t - s.start:.0f}s]: {txt}" for si, (t, txt) in enumerate(sents)]
        after = " ".join(txt for _, txt in
                         _sentences_in(transcript, s.end, s.end + _LOOK_AHEAD_S))
        blocks.append(f"CLIP {ci} (currently {s.end - s.start:.0f}s):\n" + "\n".join(lines)
                      + (f"\n  AFTER-END: {after}" if after else ""))

    resp = await client.chat.completions.create(
        model=settings.openai_analysis_model,
        response_format={"type": "json_object"}, temperature=0.2, max_tokens=600,
        messages=[{"role": "system", "content": _ENDING_PROMPT},
                  {"role": "user", "content": "\n\n".join(blocks)}])
    verdicts = {int(v.get("i", -1)): v
                for v in json.loads(resp.choices[0].message.content or "{}").get("endings", [])}

    out: list[Segment] = []
    for ci, s in enumerate(segs):
        v = verdicts.get(ci)
        sents = per_clip_sentences[ci]
        if not v or not sents:
            out.append(s)
            continue
        try:
            idx = int(v.get("end_sentence", len(sents) - 1))
            conf = float(v.get("confidence", 0.0))
        except (TypeError, ValueError):
            out.append(s)
            continue
        if conf < _CONFIDENCE_THRESHOLD or not (0 <= idx < len(sents)):
            out.append(s)
            continue
        new_end = sents[idx][0]
        # Only trim; respect the fragment floor; ignore noise-level trims.
        if new_end >= s.end - _MIN_TRIM_S or (new_end - s.start) < hard_floor:
            out.append(s)
            continue
        logger.info("ending detector: clip {} trimmed {:.0f}s → {:.0f}s ({})",
                    ci, s.end - s.start, new_end - s.start, v.get("reason", ""))
        out.append(s.model_copy(update={"end": round(new_end, 2)}))
    return out
