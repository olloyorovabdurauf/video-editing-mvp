"""
Reels script generator.

Given a topic + language + style, produce a complete, retention-engineered
Instagram Reels script as strict JSON. The whole point is to fix the failure
mode of thin 15s clips: every script is sized for 40-60s, follows a fixed
hook→problem→value→payoff arc, and ALWAYS pays off the hook's promise.

Design notes
------------
- Native-language generation, not translation. For Uzbek we instruct the model
  to *think* in Uzbek (Latin script), conversational creator tone.
- We validate the structure (all four beats present, duration floor, hook is
  short) and retry once with a sharper instruction before giving up — the same
  "never hand back something broken" stance as the segment picker.
- `_call_llm` is the only network seam, so tests can stub it without OpenAI.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.script import (
    REQUIRED_SECTIONS,
    Caption,
    ScriptGenerateRequest,
    ScriptLanguage,
    ScriptResponse,
    ScriptSection,
    ScriptStyle,
)
from app.services import ai_cache

# Words a presenter speaks per second at a natural Reels pace. Used to size the
# script so it actually fills the target duration (under-writing = thin reels).
_WORDS_PER_SECOND = 2.6


class ScriptGenerationError(RuntimeError):
    """The model failed to return a usable script after a retry."""


_LANGUAGE_RULES: dict[ScriptLanguage, str] = {
    ScriptLanguage.EN: (
        "Write EVERYTHING (title, hook, voiceover, visuals, caption, hashtags) in "
        "natural, conversational English."
    ),
    ScriptLanguage.RU: (
        "Пиши ВСЁ (заголовок, хук, озвучку, визуал, подпись, хэштеги) на "
        "естественном разговорном русском языке. Не перевод — думай по-русски."
    ),
    ScriptLanguage.UZ: (
        "HAMMASINI (sarlavha, hook, voiceover, vizual, izoh/caption, hashtaglar) "
        "tabiiy, jonli O'ZBEK tilida lotin alifbosida yoz. BU TARJIMA EMAS — "
        "o'zbekcha fikrla, o'zbek creatorlar gapiradigan suhbat ohangida. "
        "G'aliz, so'zma-so'z tarjima qilingan jumlalardan qoch."
    ),
}

_STYLE_RULES: dict[ScriptStyle, str] = {
    ScriptStyle.FOUNDER: (
        "STYLE = founder / build-in-public. Speak as a startup founder sharing the "
        "journey: building in public, the AI future, hard startup lessons, real "
        "product-building details. Be specific and a little vulnerable — concrete "
        "numbers, real decisions, named trade-offs. No generic motivation."
    ),
    ScriptStyle.EDUCATIONAL: (
        "STYLE = educational. Teach one idea using: Problem → Insight → Framework "
        "→ Example. The viewer should be able to act on it immediately. Prefer one "
        "sharp framework over many shallow tips."
    ),
}

# Few-shot anchor in Uzbek, straight from the product spec, so the model matches
# the desired caption voice for the most important launch language.
_UZ_CAPTION_EXAMPLE = (
    "Caption namunasi (uslub uchun, nusxa ko'chirma):\n"
    "  hook: \"Video editing 3 soat vaqt olayaptimi?\"\n"
    "  body: \"Creatorlarning eng katta muammosi — montaj. AI buni necha daqiqaga "
    "qisqartiradi va nimani avtomatlashtiradi...\"\n"
    "  cta: \"AI video editing kelajagini kuzatib boring.\""
)

SYSTEM_PROMPT = """\
You are a world-class short-form content strategist, viral scriptwriter, and
retention specialist. You write Instagram Reels scripts that hook in the first
second AND hold watch-time to the very end. You optimize for retention, watch
time, saves, and shares.

NON-NEGOTIABLE STRUCTURE (timestamps are targets, total length {duration}s):
  0-5s   HOOK     — pattern interrupt + curiosity. Viewer must think "I need this".
  5-15s  PROBLEM  — why it matters; make the audience relate.
  15-{value_end}s VALUE — develop the idea deeply: steps, examples, story, insight.
  {value_end}-{duration}s PAYOFF + CTA — close the loop, answer the opening
           curiosity explicitly, then one clear next action.

HARD RULES:
- NEVER clickbait without delivery. The hook makes a promise; the PAYOFF must
  complete it. Bad: "AI will replace everyone..." (no follow-through). Good:
  "AI will replace some editing tasks — here are the 3 disappearing first..."
  then actually name the 3.
- Fully develop the idea. Total voiceover ≈ {target_words} words so it genuinely
  fills {duration}s at a natural pace. Do NOT under-write — thin scripts are the
  failure we are fixing.
- Re-hook every ~7s (open loop, mini-cliffhanger, or a "but here's the thing")
  so people don't drop off.
- Be concrete and specific. Numbers, names, real examples beat vague claims.

{style_rule}

{language_rule}
{language_example}

OUTPUT: return STRICT JSON only (no prose, no markdown). Shape:
{{
  "title": "<scroll-stopping title>",
  "hashtags": ["<5-8 relevant tags, no # needed>"],
  "sections": [
    {{"name":"hook","start_s":0,"end_s":5,"voiceover":"<spoken words>","visual":"<on-screen / b-roll>"}},
    {{"name":"problem","start_s":5,"end_s":15,"voiceover":"...","visual":"..."}},
    {{"name":"value","start_s":15,"end_s":{value_end},"voiceover":"...","visual":"..."}},
    {{"name":"payoff","start_s":{value_end},"end_s":{duration},"voiceover":"...","visual":"..."}}
  ],
  "caption": {{
    "hook":"<strong first line>",
    "body":"<short paragraphs: why it matters + value summary>",
    "cta":"<clear call to action>"
  }}
}}
All four sections are REQUIRED, in order, and must reach at least 40s total.
"""

USER_TEMPLATE = """\
Topic: {topic}
Niche/context: {niche}
Platform: {platform}
Target duration: {duration} seconds (minimum 40).

Write the script now as strict JSON.
"""


def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


async def _call_llm(messages: list[dict], *, model: str, max_tokens: int) -> str:
    """Single network seam (stubbed in tests). Returns raw JSON content string."""
    client = _make_client()
    resp = await client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.8,                       # creative, but structure is enforced downstream
        max_tokens=max_tokens,
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()


def _build_messages(req: ScriptGenerateRequest) -> list[dict]:
    duration = req.duration_seconds
    value_end = max(40, duration - 15)          # payoff gets the last ~15s
    target_words = int(duration * _WORDS_PER_SECOND)
    system = SYSTEM_PROMPT.format(
        duration=duration,
        value_end=value_end,
        target_words=target_words,
        style_rule=_STYLE_RULES[req.style],
        language_rule=_LANGUAGE_RULES[req.language],
        language_example=_UZ_CAPTION_EXAMPLE if req.language is ScriptLanguage.UZ else "",
    )
    user = USER_TEMPLATE.format(
        topic=req.topic or "(none — propose a strong, specific on-brand topic for the niche)",
        niche=req.niche,
        platform=req.platform,
        duration=duration,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse(content: str) -> dict | None:
    """Defensive JSON parse — a malformed response shouldn't crash the request."""
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("script JSON parse failed ({}); first 120 chars: {!r}", e, content[:120])
        return None


def _coerce_sections(raw: Any) -> list[ScriptSection]:
    out: list[ScriptSection] = []
    for s in raw or []:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if name not in REQUIRED_SECTIONS:
            continue
        try:
            out.append(ScriptSection(
                name=name,
                start_s=float(s.get("start_s", 0)),
                end_s=float(s.get("end_s", 0)),
                voiceover=str(s.get("voiceover", "")).strip(),
                visual=str(s.get("visual", "")).strip(),
            ))
        except (TypeError, ValueError) as e:
            logger.warning("dropping malformed script section {}: {}", s, e)
    # Keep canonical order regardless of what the model emitted.
    order = {n: i for i, n in enumerate(REQUIRED_SECTIONS)}
    out.sort(key=lambda x: order[x.name])
    return out


def _validation_error(sections: list[ScriptSection], min_duration: int = 40) -> str | None:
    """Return a human reason the draft is unusable, or None if it passes."""
    names = {s.name for s in sections}
    missing = [n for n in REQUIRED_SECTIONS if n not in names]
    if missing:
        return f"missing sections: {', '.join(missing)}"
    if any(not s.voiceover for s in sections):
        return "a section has empty voiceover"
    total = max(s.end_s for s in sections)
    if total < min_duration:
        return f"too short ({total:.0f}s < {min_duration}s) — idea not fully developed"
    hook = next(s for s in sections if s.name == "hook")
    if hook.end_s > 8:
        return f"hook too long ({hook.end_s:.0f}s) — must grab in the first ~5s"
    return None


def _assemble(req: ScriptGenerateRequest, data: dict, sections: list[ScriptSection]) -> ScriptResponse:
    cap = data.get("caption") or {}
    hook = next(s for s in sections if s.name == "hook")
    caption = Caption(
        hook=str(cap.get("hook") or hook.voiceover).strip(),
        body=str(cap.get("body", "")).strip(),
        cta=str(cap.get("cta", "")).strip(),
    )
    hashtags = [str(h).lstrip("#") for h in (data.get("hashtags") or []) if str(h).strip()][:10]
    full_script = "\n\n".join(s.voiceover for s in sections)
    return ScriptResponse(
        title=str(data.get("title", "")).strip() or (req.topic or "Untitled reel"),
        hook=hook.voiceover,
        script=full_script,
        sections=sections,
        caption=caption,
        hashtags=hashtags,
        language=req.language,
        style=req.style,
        duration_seconds=req.duration_seconds,
    )


async def generate(req: ScriptGenerateRequest) -> ScriptResponse:
    """
    Generate one retention-structured script. Validates the 4-beat structure +
    duration floor; retries once with a corrective nudge; raises
    ScriptGenerationError only if the model fails twice.
    """
    settings = get_settings()
    model = settings.openai_script_model

    ck = ai_cache.key("script", req.language.value, req.style.value,
                      req.duration_seconds, req.niche, req.topic or "")
    cached = ai_cache.get_json(ck)
    if cached is not None:
        logger.info("script cache hit")
        return ScriptResponse(**cached)

    messages = _build_messages(req)
    last_reason = "no response"

    for attempt in (1, 2):
        try:
            content = await _call_llm(messages, model=model, max_tokens=2000)
        except Exception as e:                                  # network / API error
            last_reason = f"LLM call failed: {e}"
            logger.warning("script generation attempt {} errored: {}", attempt, e)
            continue

        data = _parse(content)
        if data is None:
            last_reason = "unparseable JSON"
        else:
            sections = _coerce_sections(data.get("sections"))
            reason = _validation_error(sections)
            if reason is None:
                result = _assemble(req, data, sections)
                ai_cache.set_json(ck, result.model_dump(), ttl_s=settings.script_cache_ttl_s)
                return result
            last_reason = reason

        logger.info("script attempt {} rejected: {}", attempt, last_reason)
        # Sharpen the instruction for the retry.
        messages = messages + [
            {"role": "system", "content":
                f"Your previous draft was rejected: {last_reason}. Fix it. Return ALL four "
                f"sections (hook, problem, value, payoff) in order, fully developed, totalling "
                f"at least 40 seconds, hook within 5s, and PAY OFF the hook in the payoff."}
        ]

    raise ScriptGenerationError(f"could not generate a valid script: {last_reason}")
