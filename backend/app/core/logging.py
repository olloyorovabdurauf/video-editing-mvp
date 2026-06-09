"""Structured logging. Loguru sink → stdout in dev, JSON in prod."""
import sys

from loguru import logger

from app.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    logger.remove()
    if settings.is_prod:
        logger.add(sys.stdout, serialize=True, level=settings.log_level)
    else:
        logger.add(
            sys.stdout,
            level=settings.log_level,
            format="<green>{time:HH:mm:ss}</green> "
            "<level>{level:<7}</level> "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | {message}",
        )
