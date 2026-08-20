"""Parses local `harnyx-miner-local-eval` / `harnyx-miner-local-benchmark` JSON
reports into a normalized time series, keyed off the schemas produced by
`harnyx_miner.local_eval` and `harnyx_miner.local_benchmark`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LocalEvalRun:
    path: Path
    generated_at: float
    batch_id: str | None
    mode: str | None
    target_uid: Any
    target_avg_score: float | None
    target_total_cost_usd: float | None
    champion_avg_score: float | None
    champion_total_cost_usd: float | None
    wins: int | None
    losses: int | None
    ties: int | None


@dataclass(frozen=True, slots=True)
class LocalBenchmarkRun:
    path: Path
    generated_at: float
    suite_slug: str | None
    source_batch_id: str | None
    target_uid: Any
    mean_total_score: float | None
    total_cost_usd: float | None
    item_count: int | None
    error_count: int | None


def scan_local_eval_reports(reports_dir: Path) -> list[LocalEvalRun]:
    if not reports_dir.is_dir():
        return []
    runs = [_parse_local_eval_report(path) for path in sorted(reports_dir.glob("local-eval-report-*.json"))]
    return sorted((run for run in runs if run is not None), key=lambda run: run.generated_at)


def scan_local_benchmark_reports(reports_dir: Path) -> list[LocalBenchmarkRun]:
    if not reports_dir.is_dir():
        return []
    runs = [
        _parse_local_benchmark_report(path)
        for path in sorted(reports_dir.glob("local-benchmark-report-*.json"))
    ]
    return sorted((run for run in runs if run is not None), key=lambda run: run.generated_at)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_local_eval_report(path: Path) -> LocalEvalRun | None:
    payload = _load_json(path)
    if payload is None:
        return None
    identifiers = payload.get("identifiers") or {}
    result_summary = payload.get("local_result_summary") or {}
    leaderboard = result_summary.get("leaderboard") or []
    target_entry = _find_leaderboard_entry(leaderboard, identifiers.get("target_artifact_id"))
    champion_entry = _find_leaderboard_entry(leaderboard, identifiers.get("champion_artifact_id"))
    head_to_head = result_summary.get("head_to_head") or {}
    return LocalEvalRun(
        path=path,
        generated_at=path.stat().st_mtime,
        batch_id=_as_str(identifiers.get("batch_id")),
        mode=_as_str(payload.get("mode")),
        target_uid=identifiers.get("target_uid"),
        target_avg_score=_as_float(_get(target_entry, "avg_score")),
        target_total_cost_usd=_as_float(_get(_get(target_entry, "cost_totals") or {}, "total_cost_usd")),
        champion_avg_score=_as_float(_get(champion_entry, "avg_score")),
        champion_total_cost_usd=_as_float(
            _get(_get(champion_entry, "cost_totals") or {}, "total_cost_usd")
        ),
        wins=_as_int(head_to_head.get("wins")),
        losses=_as_int(head_to_head.get("losses")),
        ties=_as_int(head_to_head.get("ties")),
    )


def _parse_local_benchmark_report(path: Path) -> LocalBenchmarkRun | None:
    payload = _load_json(path)
    if payload is None:
        return None
    identifiers = payload.get("identifiers") or {}
    summary = payload.get("summary") or {}
    manifest = (payload.get("benchmark_metadata") or {}).get("manifest") or {}
    cost_totals = summary.get("cost_totals") or {}
    return LocalBenchmarkRun(
        path=path,
        generated_at=path.stat().st_mtime,
        suite_slug=_as_str(manifest.get("suite_slug")),
        source_batch_id=_as_str(identifiers.get("source_batch_id")),
        target_uid=identifiers.get("target_uid"),
        mean_total_score=_as_float(summary.get("mean_total_score")),
        total_cost_usd=_as_float(cost_totals.get("total_cost_usd")),
        item_count=_as_int(summary.get("item_count")),
        error_count=_as_int(summary.get("error_count")),
    )


def _find_leaderboard_entry(leaderboard: Any, artifact_id: Any) -> dict[str, Any] | None:
    if artifact_id is None or not isinstance(leaderboard, list):
        return None
    for entry in leaderboard:
        if isinstance(entry, dict) and entry.get("artifact_id") == artifact_id:
            return entry
    return None


def _get(mapping: Any, key: str) -> Any:
    return mapping.get(key) if isinstance(mapping, dict) else None


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
