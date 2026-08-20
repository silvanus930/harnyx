from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from harnyx_dashboard.app import create_app
from harnyx_dashboard.config import load_settings
from harnyx_dashboard.platform_data import PlatformDataService
from harnyx_miner.platform_monitoring import PlatformMonitoringClient


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = load_settings(
        reports_dir=args.reports_dir,
        host=args.host,
        port=args.port,
        cache_seconds=args.cache_seconds,
        uid=args.uid,
        hotkey=args.hotkey,
    )
    client = PlatformMonitoringClient(base_url=settings.platform_base_url)
    platform_data = PlatformDataService(client, cache_seconds=settings.cache_seconds)
    app = create_app(settings, platform_data)
    try:
        uvicorn.run(app, host=settings.host, port=settings.port)
    finally:
        platform_data.close()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harnyx miner operator dashboard")
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory to scan for local-eval-report-*.json / local-benchmark-report-*.json files (default: cwd)",
    )
    parser.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 8787)")
    parser.add_argument(
        "--cache-seconds",
        type=float,
        default=None,
        help="How long to cache public monitoring API responses (default: 60)",
    )
    parser.add_argument("--uid", default=None, help="Your miner UID, surfaced on the hotkey-standing card")
    parser.add_argument("--hotkey", default=None, help="Your miner hotkey ss58, surfaced on the hotkey-standing card")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
