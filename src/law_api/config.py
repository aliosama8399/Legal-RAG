"""Backward-compatible shim: configuration lives in ``law_api.helpers.config``."""

from .helpers.config import PROJECT_ROOT, Settings, get_settings, settings

__all__ = ["PROJECT_ROOT", "Settings", "get_settings", "settings"]
