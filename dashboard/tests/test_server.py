from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from harnyx_dashboard.app import create_app
from harnyx_dashboard.config import Settings
from harnyx_dashboard.platform_data import PlatformDataService

_LOCAL_EVAL_REPORT = {
    "mode": "vs-champion",
    "identifiers": {
        "batch_id": "batch-1",
        "target_artifact_id": "artifact-target",
        "target_uid": 42,
        "champion_artifact_id": "artifact-champion",
        "champion_uid": 186,
    },
    "local_result_summary": {
        "leaderboard": [
            {
                "label": "target",
                "artifact_id": "artifact-target",
                "uid": 42,
                "avg_score": 0.62,
                "cost_totals": {"total_cost_usd": 0.031},
            }
        ],
        "head_to_head": {"wins": 1, "losses": 0, "ties": 0},
    },
}


class _FakePlatformDataService(PlatformDataService):
    def __init__(self) -> None:
        # Deliberately skips the base __init__: this double overrides every
        # method that would otherwise need a real PlatformMonitoringClient.
        pass

    def get_champion_overview(self) -> dict[str, object] | None:
        return {"batch_id": "batch-9", "status": "completed", "champion_artifact_id": "a-9", "champion_uid": 186}

    def list_recent_batches(self, *, limit: int = 15) -> list[dict[str, object]]:
        return [
            {"batch_id": "batch-9", "status": "completed", "champion_artifact_id": "a-9", "champion_uid": 186},
        ]

    def close(self) -> None:
        pass


def _settings(reports_dir: Path) -> Settings:
    return Settings(
        platform_base_url="https://example.test",
        reports_dir=reports_dir,
        host="127.0.0.1",
        port=8787,
        cache_seconds=60.0,
        recent_batch_limit=15,
        uid=None,
        hotkey=None,
    )


def test_index_serves_dashboard_html(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), _FakePlatformDataService())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Harnyx Miner Dashboard" in response.text


def test_overview_endpoint_combines_champion_and_latest_local_run(tmp_path: Path) -> None:
    (tmp_path / "local-eval-report-batch-1-vs-champion.json").write_text(json.dumps(_LOCAL_EVAL_REPORT))
    app = create_app(_settings(tmp_path), _FakePlatformDataService())
    client = TestClient(app)

    response = client.get("/api/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["champion"]["champion_uid"] == 186
    assert payload["local_run_count"] == 1
    assert payload["latest_local_eval"]["target_avg_score"] == 0.62
    assert payload["hotkey_standing"]["available"] is False


def test_local_eval_history_endpoint_returns_parsed_runs(tmp_path: Path) -> None:
    (tmp_path / "local-eval-report-batch-1-vs-champion.json").write_text(json.dumps(_LOCAL_EVAL_REPORT))
    app = create_app(_settings(tmp_path), _FakePlatformDataService())
    client = TestClient(app)

    response = client.get("/api/local-eval-history")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["local_eval_runs"]) == 1
    assert payload["local_eval_runs"][0]["wins"] == 1


def test_batches_endpoint_returns_recent_batches(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), _FakePlatformDataService())
    client = TestClient(app)

    response = client.get("/api/batches")

    assert response.status_code == 200
    payload = response.json()
    assert payload["batches"][0]["champion_uid"] == 186
