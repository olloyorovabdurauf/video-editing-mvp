#!/usr/bin/env bash
# Idempotent first-deploy script. Run from repo root:
#   bash deploy/fly/deploy.sh
#
# What this does:
#   1. Creates the two Fly apps if they don't exist
#   2. Provisions volumes (10GB api, 20GB worker)
#   3. Loads secrets from .env.production
#   4. Deploys both apps
#
# What it does NOT do:
#   - Set up Redis (use Upstash; paste the URL into .env.production)
#   - Set up Stripe (do that in the Stripe dashboard)
#   - Set up DNS (do that with your registrar)

set -euo pipefail

REGION="${FLY_REGION:-ord}"
ENV_FILE="${ENV_FILE:-.env.production}"

if ! command -v flyctl >/dev/null; then
  echo "flyctl not found. Install: curl -L https://fly.io/install.sh | sh"
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing. Copy backend/.env.example, fill prod values."
  exit 1
fi

# ----- Create apps (idempotent) -----
flyctl apps create reelforge-api    --org personal 2>/dev/null || true
flyctl apps create reelforge-worker --org personal 2>/dev/null || true

# ----- Create volumes (idempotent) -----
flyctl volumes list -a reelforge-api    | grep -q reelforge_storage        || \
  flyctl volumes create reelforge_storage        --region "$REGION" --size 10 -a reelforge-api
flyctl volumes list -a reelforge-worker | grep -q reelforge_worker_storage || \
  flyctl volumes create reelforge_worker_storage --region "$REGION" --size 20 -a reelforge-worker

# ----- Load secrets into both apps -----
echo "Loading secrets from $ENV_FILE..."
# shellcheck disable=SC2046
flyctl secrets set $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs) -a reelforge-api
# shellcheck disable=SC2046
flyctl secrets set $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs) -a reelforge-worker

# ----- Deploy -----
flyctl deploy -c deploy/fly/fly.api.toml    --remote-only -a reelforge-api
flyctl deploy -c deploy/fly/fly.worker.toml --remote-only -a reelforge-worker

echo
echo "✓ Deployed. Smoke test:"
echo "  curl https://reelforge-api.fly.dev/api/v1/livez"
echo "  curl https://reelforge-api.fly.dev/api/v1/healthz"
