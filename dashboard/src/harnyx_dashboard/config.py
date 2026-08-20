from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from harnyx_miner.env import load_public_env
from harnyx_miner.platform_monitoring import platform_base_url_from_env

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8787
_DEFAULT_CACHE_SECONDS = 60.0
_DEFAULT_RECENT_BATCH_LIMIT = 15


@dataclass(frozen=True, slots=True)
class Settings:
    platform_base_url: str
    reports_dir: Path
    host: str
    port: int
    cache_seconds: float
    recent_batch_limit: int
    uid: str | None
    hotkey: str | None


def load_settings(
    *,
    reports_dir: str | None = None,
    host: str | None = None,
    port: int | None = None,
    cache_seconds: float | None = None,
    uid: str | None = None,
    hotkey: str | None = None,
) -> Settings:
    load_public_env()
    return Settings(
        platform_base_url=platform_base_url_from_env(),
        reports_dir=Path(reports_dir or os.getenv("DASHBOARD_REPORTS_DIR") or ".").resolve(),
        host=host or os.getenv("DASHBOARD_HOST") or _DEFAULT_HOST,
        port=int(port or os.getenv("DASHBOARD_PORT") or _DEFAULT_PORT),
        cache_seconds=float(
            cache_seconds or os.getenv("DASHBOARD_CACHE_SECONDS") or _DEFAULT_CACHE_SECONDS
        ),
        recent_batch_limit=_DEFAULT_RECENT_BATCH_LIMIT,
        uid=uid or os.getenv("DASHBOARD_UID"),
        hotkey=hotkey or os.getenv("DASHBOARD_HOTKEY"),
    )
