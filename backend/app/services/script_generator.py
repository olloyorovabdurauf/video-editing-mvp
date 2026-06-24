"""
Reels script generator.

Given a topic + language + content type + industry, produce a complete,
retention-engineered Instagram Reels script as strict JSON. The point is to fix
thin 15s clips: every script is sized for its target duration (30/45/60s),
follows a hook→problem→value→payoff arc, and ALWAYS pays off the hook.

Voice, goal, audience and tone ADAPT to the selected content type — the product
serves creators, experts, educators, coaches, businesses and marketers, not just
founders. Founder/build-in-public is one option among many, used only when the
user picks it.

`_call_llm` is the only network seam, so tests can stub it without OpenAI.
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
    ContentType,
    Industry,
    ScriptGenerateRequest,
    ScriptLanguage,
    ScriptResponse,
    ScriptSection,
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

# Each content type carries its OWN voice, goal, audience framing and tone.
# Selecting a type is what sets the personality — we never default to "founder".
_CONTENT_TYPE_RULES: dict[ContentType, str] = {
    ContentType.EDUCATIONAL: (
        "CONTENT TYPE = educational. Goal: teach one idea the viewer can act on "
        "immediately. Structure the value as Problem → Insight → Framework → "
        "Example. Tone: clear, credible, generous. Audience: learners in this "
        "industry. Speak as a knowledgeable expert, NOT necessarily a founder."
    ),
    ContentType.PERSONAL_BRAND: (
        "CONTENT TYPE = personal brand. Goal: build the creator's authority and "
        "point of view. Share a strong, specific opinion or lesson from real "
        "experience. Tone: confident, authentic, first-person. Audience: people "
        "who follow this creator's niche. This is about expertise, not company "
        "building."
    ),
    ContentType.FOUNDER_STORY: (
        "CONTENT TYPE = founder / build-in-public (ONLY because the user selected "
        "it). Goal: bring people into the startup journey. Share concrete, a "
        "little vulnerable details — real numbers, decisions, trade-offs. Tone: "
        "candid, behind-the-scenes, first-person founder."
    ),
    ContentType.PRODUCT_MARKETING: (
        "CONTENT TYPE = product marketing. Goal: make the audience want a product "
        "or feature. Lead with the pain, show the transformation, give proof. "
        "Tone: benefit-driven and credible, NOT hypey. Audience: potential users."
    ),
    ContentType.STORYTELLING: (
        "CONTENT TYPE = storytelling. Goal: hold attention with a narrative arc — "
        "tension then resolution. Use a relatable character/moment, stakes, and a "
        "turn. Tone: vivid, emotional, cinematic. The 'value' beat is the story's "
        "rising action; the 'payoff' is its lesson."
    ),
    ContentType.TUTORIAL: (
        "CONTENT TYPE = tutorial / how-to. Goal: the viewer can DO the thing after "
        "watching. The value beat is clear numbered steps. Tone: practical, "
        "encouraging, no fluff. Audience: someone trying to accomplish this task."
    ),
    ContentType.SALES: (
        "CONTENT TYPE = sales. Goal: drive ONE specific conversion. Name the pain, "
        "present the offer, handle the top objection, add gentle urgency, end on a "
        "direct CTA. Tone: persuasive but honest — no false claims. Audience: a "
        "ready-to-buy prospect."
    ),
    ContentType.VIRAL_REEL: (
        "CONTENT TYPE = viral reel. Goal: maximize shares + saves. Bold hook, "
        "pattern interrupts, fast pace, a surprising or contrarian angle, a line "
        "people want to send to a friend. Tone: high-energy and punchy. Still pay "
        "off the hook — viral without delivery just gets unfollows."
    ),
}

_INDUSTRY_HINT: dict[Industry, str] = {
    Industry.GENERAL: "a general audience",
    Industry.BUSINESS: "business owners, operators and entrepreneurs",
    Industry.HEALTH: "health, fitness and wellness audiences (avoid medical claims)",
    Industry.EDUCATION: "students, educators and lifelong learners",
    Industry.FINANCE: "personal-finance and investing audiences (no financial advice claims)",
    Industry.TECHNOLOGY: "tech-savvy builders, engineers and early adopters",
    Industry.OTHER: "this creator's specific niche",
}

# Few-shot anchor for Uzbek CAPTION FORMAT/TONE only — not the topic. It shows
# the desired punchy first line + short body + CTA shape for the launch language.
_UZ_CAPTION_EXAMPLE = (
    "Caption FORMATI uchun namuna (faqat uslub — mavzuni foydalanuvchi mavzusiga moslang):\n"
    "  hook: \"<kuchli birinchi qator, savol yoki dadil da'vo>\"\n"
    "  body: \"<qisqa, qiymatli xulosa — nega muhim>\"\n"
    "  cta: \"<aniq harakatga chaqiruv>\""
)

SYSTEM_PROMPT = """\
You are a world-class short-form content strategist, viral scriptwriter, and
retention specialist. You write Instagram Reels scripts that hook in the first
second AND hold watch-time to the very end. You optimize for retention, watch
time, saves, and shares.

NON-NEGOTIABLE STRUCTURE (timestamps are targets, total length {duration}s):
  0-{hook_end}s    HOOK    — pattern interrupt + curiosity. Viewer must think "I need this".
  {hook_end}-{problem_end}s  PROBLEM — why it matters; make the audience relate.
  {problem_end}-{value_end}s VALUE  — develop the idea deeply: steps, examples, story, insight.
  {value_end}-{duration}s PAYOFF + CTA — close the loop, answer the opening
           curiosity explicitly, then one clear next action.

HARD RULES:
- NEVER clickbait without delivery. The hook makes a promise; the PAYOFF must
  complete it. Bad: "AI will replace everyone..." (no follow-through). Good:
  "AI will replace some editing tasks — here are the 3 disappearing first..."
  then actually name the 3.
- Fully develop the idea. Total voiceover ≈ {target_words} words so it genuinely
  fills {duration}s at a natural pace. Do NOT under-write.
- Re-hook every ~7s (open loop, mini-cliffhanger, "but here's the thing").
- Be concrete and specific. Numbers, names, real examples beat vague claims.

MATCH THE BRIEF — the script MUST fit all of:
  Industry/audience: {industry}
  Goal:    {goal}
  Audience:{audience}
  Tone:    {tone}

{content_rule}

{language_rule}
{language_example}

OUTPUT: return STRICT JSON only (no prose, no markdown). Shape:
{{
  "title": "<scroll-stopping title>",
  "hashtags": ["<5-8 relevant tags, no # needed>"],
  "sections": [
    {{"name":"hook","start_s":0,"end_s":{hook_end},"voiceover":"<spoken words>","visual":"<on-screen / b-roll>"}},
    {{"name":"problem","start_s":{hook_end},"end_s":{problem_end},"voiceover":"...","visual":"..."}},
    {{"name":"value","start_s":{problem_end},"end_s":{value_end},"voiceover":"...","visual":"..."}},
    {{"name":"payoff","start_s":{value_end},"end_s":{duration},"voiceover":"...","visual":"..."}}
  ],
  "caption": {{
    "hook":"<strong first line>",
    "body":"<short paragraphs: why it matters + value summary>",
    "cta":"<clear call to action>"
  }}
}}
All four sections are REQUIRED, in order, and must reach at least {min_total}s total.
"""

USER_TEMPLATE = """\
Topic: {topic}
Industry: {industry}
Platform: {platform}
Target duration: {duration} seconds.

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


def _timeline(duration: int) -> tuple[int, int, int]:
    """Section boundaries scaled to the target duration (works for 30/45/60s)."""
    hook_end = 5 if duration >= 40 else 4
    problem_end = max(hook_end + 4, round(duration * 0.25))
    value_end = max(problem_end + 6, round(duration * 0.75))
    return hook_end, problem_end, value_end


def _min_total(duration: int) -> int:
    """Floor the model must reach so the idea is fully developed (80% of target)."""
    return max(24, round(duration * 0.8))


def _default_goal(ct: ContentType) -> str:
    return {
        ContentType.EDUCATIONAL: "teach one actionable idea",
        ContentType.PERSONAL_BRAND: "build the creator's authority",
        ContentType.FOUNDER_STORY: "share the build-in-public journey",
        ContentType.PRODUCT_MARKETING: "create desire for the product",
        ContentType.STORYTELLING: "hold attention with a story and land a lesson",
        ContentType.TUTORIAL: "teach the viewer to do it themselves",
        ContentType.SALES: "drive one specific conversion",
        ContentType.VIRAL_REEL: "maximize shares and saves",
    }[ct]


def _build_messages(req: ScriptGenerateRequest) -> list[dict]:
    duration = req.duration_seconds
    hook_end, problem_end, value_end = _timeline(duration)
    target_words = int(duration * _WORDS_PER_SECOND)
    system = SYSTEM_PROMPT.format(
        duration=duration,
        hook_end=hook_end,
        problem_end=problem_end,
        value_end=value_end,
        min_total=_min_total(duration),
        target_words=target_words,
        industry=_INDUSTRY_HINT[req.industry],
        goal=req.goal or _default_goal(req.content_type),
        audience=req.audience or "infer from the content type, industry and topic",
        tone=req.tone or "infer the most effective tone for this content type",
        content_rule=_CONTENT_TYPE_RULES[req.content_type],
        language_rule=_LANGUAGE_RULES[req.language],
        language_example=_UZ_CAPTION_EXAMPLE if req.language is ScriptLanguage.UZ else "",
    )
    user = USER_TEMPLATE.format(
        topic=req.topic or "(none — propose a strong, specific topic for this content type + industry)",
        industry=req.industry.value,
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


def _validation_error(sections: list[ScriptSection], *, min_total: int, hook_max: int) -> str | None:
    """Return a human reason the draft is unusable, or None if it passes."""
    names = {s.name for s in sections}
    missing = [n for n in REQUIRED_SECTIONS if n not in names]
    if missing:
        return f"missing sections: {', '.join(missing)}"
    if any(not s.voiceover for s in sections):
        return "a section has empty voiceover"
    total = max(s.end_s for s in sections)
    if total < min_total:
        return f"too short ({total:.0f}s < {min_total}s) — idea not fully developed"
    hook = next(s for s in sections if s.name == "hook")
    if hook.end_s > hook_max:
        return f"hook too long ({hook.end_s:.0f}s) — must grab in the first few seconds"
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
        content_type=req.content_type,
        industry=req.industry,
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
    hook_end, _, _ = _timeline(req.duration_seconds)
    min_total = _min_total(req.duration_seconds)

    ck = ai_cache.key("script", req.language.value, req.content_type.value,
                      req.industry.value, req.duration_seconds,
                      req.audience or "", req.goal or "", req.tone or "", req.topic or "")
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
            reason = _validation_error(sections, min_total=min_total, hook_max=hook_end + 3)
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
                f"at least {min_total} seconds, hook within {hook_end}s, and PAY OFF the hook "
                f"in the payoff."}
        ]

    raise ScriptGenerationError(f"could not generate a valid script: {last_reason}")
