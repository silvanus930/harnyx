from __future__ import annotations

from typing import Any
from uuid import UUID

from harnyx_dashboard.platform_data import PlatformDataService

_BATCH_ID = "11111111-1111-1111-1111-111111111111"
_ARTIFACT_ID = "22222222-2222-2222-2222-222222222222"


class _FakeMonitoringClient:
    def __init__(self, batches: list[dict[str, Any]], scripts: dict[str, dict[str, Any]]) -> None:
        self._batches = batches
        self._scripts = scripts
        self.list_calls = 0
        self.script_calls: list[str] = []

    def list_recent_completed_batches(self, *, limit: int) -> list[dict[str, Any]]:
        self.list_calls += 1
        return self._batches[:limit]

    def get_script(self, artifact_id: UUID) -> dict[str, Any]:
        self.script_calls.append(str(artifact_id))
        return self._scripts[str(artifact_id)]


def _batch_row(champion_artifact_id: str | None = _ARTIFACT_ID) -> dict[str, Any]:
    return {"batch_id": _BATCH_ID, "status": "completed", "champion_artifact_id": champion_artifact_id}


def test_list_recent_batches_resolves_champion_uid_via_get_script() -> None:
    client = _FakeMonitoringClient(
        batches=[_batch_row()],
        scripts={_ARTIFACT_ID: {"uid": 186, "artifact_id": _ARTIFACT_ID}},
    )
    service = PlatformDataService(client, cache_seconds=60.0)

    result = service.list_recent_batches(limit=5)

    assert result == [
        {
            "batch_id": _BATCH_ID,
            "status": "completed",
            "champion_artifact_id": _ARTIFACT_ID,
            "champion_uid": 186,
        }
    ]
    assert client.script_calls == [_ARTIFACT_ID]


def test_list_recent_batches_handles_batch_with_no_champion_yet() -> None:
    client = _FakeMonitoringClient(batches=[_batch_row(champion_artifact_id=None)], scripts={})
    service = PlatformDataService(client, cache_seconds=60.0)

    result = service.list_recent_batches(limit=5)

    assert result == [
        {"batch_id": _BATCH_ID, "status": "completed", "champion_artifact_id": None, "champion_uid": None}
    ]
    assert client.script_calls == []


def test_list_recent_batches_degrades_gracefully_when_script_lookup_fails() -> None:
    class _FailingClient(_FakeMonitoringClient):
        def get_script(self, artifact_id: UUID) -> dict[str, Any]:
            raise RuntimeError("platform unavailable")

    client = _FailingClient(batches=[_batch_row()], scripts={})
    service = PlatformDataService(client, cache_seconds=60.0)

    result = service.list_recent_batches(limit=5)

    assert result == [
        {
            "batch_id": _BATCH_ID,
            "status": "completed",
            "champion_artifact_id": _ARTIFACT_ID,
            "champion_uid": None,
        }
    ]


def test_list_recent_batches_caches_within_ttl() -> None:
    client = _FakeMonitoringClient(
        batches=[_batch_row()],
        scripts={_ARTIFACT_ID: {"uid": 186, "artifact_id": _ARTIFACT_ID}},
    )
    service = PlatformDataService(client, cache_seconds=60.0)

    service.list_recent_batches(limit=5)
    service.list_recent_batches(limit=5)

    assert client.list_calls == 1


def test_uid_resolution_is_memoized_across_batches_sharing_a_champion() -> None:
    client = _FakeMonitoringClient(
        batches=[_batch_row(), _batch_row()],
        scripts={_ARTIFACT_ID: {"uid": 186, "artifact_id": _ARTIFACT_ID}},
    )
    service = PlatformDataService(client, cache_seconds=0.0)

    service.list_recent_batches(limit=5)
    service._cache.clear()  # force a re-fetch of the batch list, bypassing only the TTL cache
    service.list_recent_batches(limit=5)

    assert client.script_calls == [_ARTIFACT_ID]


def test_get_champion_overview_returns_first_recent_batch_or_none() -> None:
    empty_client = _FakeMonitoringClient(batches=[], scripts={})
    empty_service = PlatformDataService(empty_client, cache_seconds=60.0)
    assert empty_service.get_champion_overview() is None

    client = _FakeMonitoringClient(
        batches=[_batch_row()],
        scripts={_ARTIFACT_ID: {"uid": 186, "artifact_id": _ARTIFACT_ID}},
    )
    service = PlatformDataService(client, cache_seconds=60.0)
    overview = service.get_champion_overview()
    assert overview is not None
    assert overview["champion_uid"] == 186
