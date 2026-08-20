from __future__ import annotations

import json
from pathlib import Path

from harnyx_dashboard.reports import scan_local_benchmark_reports, scan_local_eval_reports

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
            },
            {
                "label": "champion",
                "artifact_id": "artifact-champion",
                "uid": 186,
                "avg_score": 0.71,
                "cost_totals": {"total_cost_usd": 0.048},
            },
        ],
        "head_to_head": {"wins": 3, "losses": 6, "ties": 1},
    },
}

_LOCAL_BENCHMARK_REPORT = {
    "identifiers": {"source_batch_id": "batch-1", "target_uid": 42},
    "benchmark_metadata": {"manifest": {"suite_slug": "draco"}},
    "summary": {
        "mean_total_score": 0.55,
        "item_count": 20,
        "error_count": 1,
        "cost_totals": {"total_cost_usd": 0.12},
    },
}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_scan_local_eval_reports_extracts_target_and_champion_by_artifact_id(tmp_path: Path) -> None:
    _write(tmp_path / "local-eval-report-batch-1-vs-champion.json", _LOCAL_EVAL_REPORT)

    runs = scan_local_eval_reports(tmp_path)

    assert len(runs) == 1
    run = runs[0]
    assert run.batch_id == "batch-1"
    assert run.mode == "vs-champion"
    assert run.target_avg_score == 0.62
    assert run.target_total_cost_usd == 0.031
    assert run.champion_avg_score == 0.71
    assert run.champion_total_cost_usd == 0.048
    assert (run.wins, run.losses, run.ties) == (3, 6, 1)


def test_scan_local_eval_reports_sorts_by_mtime_and_skips_malformed_files(tmp_path: Path) -> None:
    older = tmp_path / "local-eval-report-batch-1-vs-champion.json"
    newer = tmp_path / "local-eval-report-batch-2-vs-champion.json"
    _write(older, _LOCAL_EVAL_REPORT)
    newer_payload = dict(_LOCAL_EVAL_REPORT)
    newer_payload["identifiers"] = {**_LOCAL_EVAL_REPORT["identifiers"], "batch_id": "batch-2"}
    _write(newer, newer_payload)
    (tmp_path / "local-eval-report-broken-vs-champion.json").write_text("not json")

    import os
    import time

    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))

    runs = scan_local_eval_reports(tmp_path)

    assert [run.batch_id for run in runs] == ["batch-1", "batch-2"]


def test_scan_local_eval_reports_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    assert scan_local_eval_reports(tmp_path / "does-not-exist") == []


def test_scan_local_benchmark_reports_extracts_summary_and_manifest(tmp_path: Path) -> None:
    _write(tmp_path / "local-benchmark-report-batch-1-v1-v1.json", _LOCAL_BENCHMARK_REPORT)

    runs = scan_local_benchmark_reports(tmp_path)

    assert len(runs) == 1
    run = runs[0]
    assert run.suite_slug == "draco"
    assert run.source_batch_id == "batch-1"
    assert run.mean_total_score == 0.55
    assert run.total_cost_usd == 0.12
    assert run.item_count == 20
    assert run.error_count == 1
