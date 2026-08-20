from __future__ import annotations

import httpx
import pytest

from harnyx_miner.platform_monitoring import PlatformMonitoringClient

_BASE_URL = "https://example.test"


def _build_client(handler: object) -> PlatformMonitoringClient:
    return PlatformMonitoringClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


def test_list_recent_completed_batches_filters_and_paginates() -> None:
    pages = [
        {
            "batches": [
                {"batch_id": "b1", "status": "completed"},
                {"batch_id": "b2", "status": "running"},
                {"batch_id": "b3", "status": "completed"},
            ],
            "next_before": "cursor-1",
            "next_before_batch_id": "b3",
        },
        {
            "batches": [
                {"batch_id": "b4", "status": "completed"},
            ],
            "next_before": None,
            "next_before_batch_id": None,
        },
    ]
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=pages[len(calls) - 1])

    client = _build_client(handler)
    result = client.list_recent_completed_batches(limit=10)

    assert [batch["batch_id"] for batch in result] == ["b1", "b3", "b4"]
    assert len(calls) == 2


def test_list_recent_completed_batches_stops_at_limit_without_extra_page() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "batches": [
                    {"batch_id": "b1", "status": "completed"},
                    {"batch_id": "b2", "status": "completed"},
                ],
                "next_before": "cursor",
                "next_before_batch_id": "b2",
            },
        )

    client = _build_client(handler)
    result = client.list_recent_completed_batches(limit=1)

    assert [batch["batch_id"] for batch in result] == ["b1"]
    assert len(calls) == 1


def test_list_recent_completed_batches_rejects_non_positive_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call expected for a non-positive limit")

    client = _build_client(handler)

    assert client.list_recent_completed_batches(limit=0) == []


def test_find_latest_completed_batch_delegates_to_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"batches": [{"batch_id": "only", "status": "completed"}], "next_before": None},
        )

    client = _build_client(handler)

    assert client.find_latest_completed_batch()["batch_id"] == "only"


def test_find_latest_completed_batch_raises_when_none_available() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"batches": [], "next_before": None})

    client = _build_client(handler)

    with pytest.raises(RuntimeError, match="no completed public miner-task batch"):
        client.find_latest_completed_batch()
