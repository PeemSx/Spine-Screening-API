"""Application logging configuration without request or patient payloads."""

from __future__ import annotations

import logging


def configure_logging(log_level: str) -> None:
    """Configure a concise process-wide default and the application logger level."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("app").setLevel(level)
