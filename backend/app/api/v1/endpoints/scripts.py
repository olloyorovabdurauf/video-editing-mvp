"""
POST /api/v1/scripts — generate a retention-structured Reels script.

Text-only product (no video pipeline): topic + language + style in, a complete
40-60s hook→problem→value→payoff script out, natively in en/ru/uz. Auth-gated
and rate-limited; not credit-charged (a single short LLM call is cheap — wire a
flat credit cost here if/when you meter it).
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import require_user
from app.core.rate_limit import rate_limit
from app.schemas.script import ScriptGenerateRequest, ScriptResponse
from app.services import script_generator

router = APIRouter(prefix="/scripts", tags=["scripts"])

_script_rate_limit = rate_limit("scripts", max_per_window=20, window_s=60)


@router.post(
    "",
    response_model=ScriptResponse,
    summary="Generate a retention-structured Reels script",
    dependencies=[Depends(_script_rate_limit)],
)
async def create_script(
    req: ScriptGenerateRequest,
    user_id: str = Depends(require_user),
) -> ScriptResponse:
    req.user_id = user_id  # trust the token, not the body
    try:
        return await script_generator.generate(req)
    except script_generator.ScriptGenerationError as e:
        # The model couldn't produce a valid script even after a retry.
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"script generation failed: {e}")
