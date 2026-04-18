"""Bulk-download OHLCV data from Massive.com API.

Usage:
    uv run python scripts/fetch_data.py --timeframe daily --start 2020-01-01
    uv run python scripts/fetch_data.py --symbols AAPL,NVDA --timeframe 5min
    uv run python scripts/fetch_data.py --universe configs/symbols.yaml
"""

from __future__ import annotations

import argparse

from aprilalgo.data.fetcher import fetch_universe
from aprilalgo.data.universe import load_universe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLCV data from Massive API")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (e.g. AAPL,NVDA,TSLA)")
    parser.add_argument("--universe", type=str, default=None, help="Path to a symbols YAML or text file")
    parser.add_argument(
        "--timeframe", type=str, default="daily", help="Timeframe: 1min, 5min, 15min, 30min, hourly, daily, weekly"
    )
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD, defaults to today)")
    parser.add_argument("--api-key", type=str, default=None, help="Massive API key (or set MASSIVE_API_KEY env var)")
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else load_universe(args.universe)

    print(f"Fetching {args.timeframe} data for {len(symbols)} symbols")
    print(f"  Range: {args.start} → {args.end or 'today'}")
    print(f"  Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    print()

    results = fetch_universe(
        symbols=symbols,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        api_key=args.api_key,
    )

    print()
    print(f"Done. Fetched {len(results)}/{len(symbols)} symbols successfully.")


if __name__ == "__main__":
    main()
