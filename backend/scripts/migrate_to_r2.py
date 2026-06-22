"""
One-time migration: copy existing reels from the Fly volume → Cloudflare R2.

There isn't much on the volume (recent renders) — the real win is the *forward*
cutover: once STORAGE_BACKEND=s3, every NEW job writes to R2. This script moves
the existing /app/storage/output/** so old links keep working.

Usage (on the Fly machine, after R2 secrets are set):
    flyctl ssh console -a reelforge-mvp-x7k2 -C "python -m scripts.migrate_to_r2"

Idempotent: re-running skips objects already in R2. Dry-run with --dry-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.config import get_settings


def main(dry_run: bool = False) -> None:
    s = get_settings()
    root = s.storage_local_dir
    if s.storage_backend.lower() != "s3":
        print("STORAGE_BACKEND is not 's3' — set R2 secrets first. Aborting.")
        sys.exit(1)

    import boto3
    client = boto3.client(
        "s3", region_name=s.s3_region or None, endpoint_url=s.s3_endpoint_url or None,
        aws_access_key_id=s.aws_access_key_id or None,
        aws_secret_access_key=s.aws_secret_access_key or None,
    )

    moved = skipped = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        key = str(path.relative_to(root)).replace("\\", "/")   # e.g. output/<job>/reel_0.mp4
        try:
            client.head_object(Bucket=s.s3_bucket, Key=key)
            skipped += 1
            continue                                            # already in R2
        except client.exceptions.ClientError:
            pass
        print(f"{'DRY ' if dry_run else ''}upload {key} ({path.stat().st_size} bytes)")
        if not dry_run:
            client.upload_file(str(path), s.s3_bucket, key)
        moved += 1

    print(f"done: {moved} uploaded, {skipped} already present")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
