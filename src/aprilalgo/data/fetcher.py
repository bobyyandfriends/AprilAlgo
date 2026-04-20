"""Fetch OHLCV data from Massive.com (formerly Polygon.io) REST API."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd

from aprilalgo.data.loader import get_price_path

_TIMESPAN_MAP = {
    "1min": (1, "minute"),
    "5min": (5, "minute"),
    "15min": (15, "minute"),
    "30min": (30, "minute"),
    "hourly": (1, "hour"),
    "daily": (1, "day"),
    "weekly": (1, "week"),
}


def fetch_bars(
    symbol: str,
    timeframe: str = "daily",
    start: str = "2020-01-01",
    end: str | None = None,
    api_key: str | None = None,
    save: bool = True,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Download OHLCV bars from Massive API and optionally save to CSV.

    Parameters
    ----------
    symbol : Ticker symbol (e.g. "AAPL").
    timeframe : One of "1min", "5min", "15min", "30min", "hourly", "daily", "weekly".
    start : Start date as YYYY-MM-DD string.
    end : End date (defaults to today).
    api_key : Massive API key. Falls back to MASSIVE_API_KEY env var if None.
    save : If True, write the result to the standard CSV path.
    data_dir : Override for the data directory root.
    """
    try:
        from massive import RESTClient
    except ImportError as exc:
        raise ImportError(
            "The 'massive' package is required for data fetching. "
            "Install it with: uv add massive"
        ) from exc

    if timeframe not in _TIMESPAN_MAP:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Choose from: {list(_TIMESPAN_MAP.keys())}"
        )

    multiplier, timespan = _TIMESPAN_MAP[timeframe]
    end = end or datetime.now().strftime("%Y-%m-%d")

    client = RESTClient(api_key=api_key) if api_key else RESTClient()

    raw_bars = []
    for bar in client.list_aggs(
        ticker=symbol.upper(),
        multiplier=multiplier,
        timespan=timespan,
        from_=start,
        to=end,
        limit=50000,
    ):
        raw_bars.append({
            "datetime": pd.Timestamp(bar.timestamp, unit="ms"),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })

    if not raw_bars:
        raise ValueError(f"No data returned for {symbol} {timeframe} ({start} to {end})")

    df = pd.DataFrame(raw_bars)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if save:
        out_path = get_price_path(symbol, timeframe, data_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df


def fetch_universe(
    symbols: list[str],
    timeframe: str = "daily",
    start: str = "2020-01-01",
    end: str | None = None,
    api_key: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch bars for multiple symbols, saving each to CSV.

    Returns a dict mapping symbol → DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = fetch_bars(
                symbol=sym,
                timeframe=timeframe,
                start=start,
                end=end,
                api_key=api_key,
                save=True,
                data_dir=data_dir,
            )
            results[sym] = df
            print(f"  {sym}: {len(df)} bars")
        except Exception as exc:
            print(f"  {sym}: FAILED — {exc}")
    return results
