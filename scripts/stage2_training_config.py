"""Compatibility wrapper for scripts/pretrain_config.py."""

try:
    from pretrain_config import *  # noqa: F401,F403
except ModuleNotFoundError:  # pragma: no cover - package import fallback.
    from scripts.pretrain_config import *  # noqa: F401,F403
