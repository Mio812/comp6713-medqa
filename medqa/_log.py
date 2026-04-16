"""Shared logging helpers for the medqa package."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``medqa.*`` namespace.

    Honours ``MEDQA_LOG_LEVEL`` (DEBUG / INFO / WARNING / ERROR), default INFO.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        level_name = os.environ.get("MEDQA_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        root = logging.getLogger("medqa")
        root.setLevel(level)
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(
                fmt="[%(name)s] %(message)s",
            ))
            root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True

    if not name.startswith("medqa"):
        name = f"medqa.{name}"
    return logging.getLogger(name)
