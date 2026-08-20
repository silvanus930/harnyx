"""Thin, cached aggregator over `PlatformMonitoringClient` (public,
unauthenticated `/v1/monitoring/...` endpoints) so browser polling doesn't
re-hit the platform API on every request.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from harnyx_miner.platform_monitoring import PlatformMonitoringClient


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expires_at: float


class PlatformDataService:
    def __init__(self, client: PlatformMonitoringClient, *, cache_seconds: float = 60.0) -> None:
        self._client = client
        self._cache_seconds = cache_seconds
        self._cache: dict[str, _CacheEntry] = {}
        # artifact_id -> uid never changes once submitted, so this is a
        # permanent memo rather than a TTL cache entry.
        self._uid_by_artifact_id: dict[str, Any] = {}

    def close(self) -> None:
        self._client.close()

    def get_champion_overview(self) -> dict[str, Any] | None:
        batches = self.list_recent_batches(limit=1)
        return batches[0] if batches else None

    def list_recent_batches(self, *, limit: int = 15) -> list[dict[str, Any]]:
        return self._cached(f"recent_batches:{limit}", lambda: self._fetch_recent_batches(limit))

    def _cached(self, key: str, loader: Callable[[], Any]) -> Any:
        now = time.monotonic()
        entry = self._cache.get(key)
        if entry is not None and entry.expires_at > now:
            return entry.value
        value = loader()
        self._cache[key] = _CacheEntry(value=value, expires_at=now + self._cache_seconds)
        return value

    def _fetch_recent_batches(self, limit: int) -> list[dict[str, Any]]:
        batches = self._client.list_recent_completed_batches(limit=limit)
        return [self._summarize_batch(batch) for batch in batches]

    def _summarize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        champion_artifact_id = batch.get("champion_artifact_id")
        return {
            "batch_id": batch.get("batch_id"),
            "status": batch.get("status"),
            "champion_artifact_id": champion_artifact_id,
            "champion_uid": self._resolve_uid(champion_artifact_id),
        }

    def _resolve_uid(self, artifact_id: Any) -> Any:
        if artifact_id is None:
            return None
        key = str(artifact_id)
        if key in self._uid_by_artifact_id:
            return self._uid_by_artifact_id[key]
        try:
            script = self._client.get_script(UUID(key))
        except Exception:
            return None
        uid = script.get("uid")
        self._uid_by_artifact_id[key] = uid
        return uid
