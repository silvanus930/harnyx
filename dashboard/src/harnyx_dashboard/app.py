from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse

from harnyx_dashboard.config import Settings
from harnyx_dashboard.platform_data import PlatformDataService
from harnyx_dashboard.reports import (
    LocalBenchmarkRun,
    LocalEvalRun,
    scan_local_benchmark_reports,
    scan_local_eval_reports,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings, platform_data: PlatformDataService) -> FastAPI:
    app = FastAPI(title="Harnyx miner dashboard")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return _build_overview(settings, platform_data)

    @app.get("/api/local-eval-history")
    def local_eval_history() -> dict[str, Any]:
        return _build_local_eval_history(settings)

    @app.get("/api/batches")
    def batches() -> dict[str, Any]:
        recent = platform_data.list_recent_batches(limit=settings.recent_batch_limit)
        return {"batches": recent}

    return app


def _build_overview(settings: Settings, platform_data: PlatformDataService) -> dict[str, Any]:
    champion = platform_data.get_champion_overview()
    eval_runs = scan_local_eval_reports(settings.reports_dir)
    benchmark_runs = scan_local_benchmark_reports(settings.reports_dir)
    latest_eval = eval_runs[-1] if eval_runs else None
    return {
        "champion": champion,
        "local_run_count": len(eval_runs) + len(benchmark_runs),
        "latest_local_eval": _eval_run_to_dict(latest_eval) if latest_eval else None,
        "hotkey_standing": {
            "configured_uid": settings.uid,
            "configured_hotkey": settings.hotkey,
            "available": False,
            "note": (
                "Hotkey-specific standing isn't resolvable from the public "
                "monitoring API alone yet -- it needs an artifact_id you already "
                "know from a submission you made. Use the Mining Runbook's MCP "
                "workflow to look up your own artifact's recent results."
            ),
        },
    }


def _build_local_eval_history(settings: Settings) -> dict[str, Any]:
    eval_runs = scan_local_eval_reports(settings.reports_dir)
    benchmark_runs = scan_local_benchmark_reports(settings.reports_dir)
    return {
        "local_eval_runs": [_eval_run_to_dict(run) for run in eval_runs],
        "local_benchmark_runs": [_benchmark_run_to_dict(run) for run in benchmark_runs],
    }


def _eval_run_to_dict(run: LocalEvalRun) -> dict[str, Any]:
    return {
        "path": run.path.name,
        "generated_at": run.generated_at,
        "batch_id": run.batch_id,
        "mode": run.mode,
        "target_uid": run.target_uid,
        "target_avg_score": run.target_avg_score,
        "target_total_cost_usd": run.target_total_cost_usd,
        "champion_avg_score": run.champion_avg_score,
        "champion_total_cost_usd": run.champion_total_cost_usd,
        "wins": run.wins,
        "losses": run.losses,
        "ties": run.ties,
    }


def _benchmark_run_to_dict(run: LocalBenchmarkRun) -> dict[str, Any]:
    return {
        "path": run.path.name,
        "generated_at": run.generated_at,
        "suite_slug": run.suite_slug,
        "source_batch_id": run.source_batch_id,
        "target_uid": run.target_uid,
        "mean_total_score": run.mean_total_score,
        "total_cost_usd": run.total_cost_usd,
        "item_count": run.item_count,
        "error_count": run.error_count,
    }
