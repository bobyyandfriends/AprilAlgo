"""AprilAlgo — CLI entry point for running backtests."""

from __future__ import annotations

import argparse
import sys

from aprilalgo.config import load_config
from aprilalgo.data import load_price_data
from aprilalgo.backtest import run_backtest
from aprilalgo.strategies import STRATEGIES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an AprilAlgo backtest")
    parser.add_argument("--symbol", type=str, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--timeframe", type=str, help="Data timeframe (e.g. daily, 5min)")
    parser.add_argument("--strategy", type=str, help="Strategy name (e.g. rsi_sma)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.symbol:
        cfg["symbol"] = args.symbol
    if args.timeframe:
        cfg["timeframe"] = args.timeframe
    if args.strategy:
        cfg["strategy"] = args.strategy

    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    strategy_name = cfg["strategy"]

    print(f"AprilAlgo Backtest")
    print(f"{'='*50}")
    print(f"Symbol:     {symbol}")
    print(f"Timeframe:  {timeframe}")
    print(f"Strategy:   {strategy_name}")
    print(f"Capital:    ${cfg['initial_capital']:,.2f}")
    print(f"Commission: ${cfg['commission']}")
    print(f"Slippage:   {cfg['slippage']*100:.2f}%")
    print()

    # Load data
    print(f"Loading {symbol} {timeframe} data...")
    price_data = load_price_data(symbol, timeframe)
    print(f"  Loaded {len(price_data)} bars  ({price_data['datetime'].iloc[0].date()} to {price_data['datetime'].iloc[-1].date()})")
    print()

    # Create strategy
    if strategy_name not in STRATEGIES:
        print(f"ERROR: Unknown strategy '{strategy_name}'. Available: {list(STRATEGIES.keys())}")
        sys.exit(1)
    strategy = STRATEGIES[strategy_name](**cfg.get("strategy_params", {}))

    # Run backtest
    print("Running backtest...")
    results = run_backtest(
        strategy=strategy,
        price_data=price_data,
        initial_capital=cfg["initial_capital"],
        commission=cfg["commission"],
        slippage=cfg["slippage"],
    )

    # Print results
    metrics = results["metrics"]
    trades_df = results["trades"]

    print()
    print(f"Results")
    print(f"{'='*50}")
    print(f"Total P&L:      ${metrics['total_pnl']:>12,.2f}")
    print(f"Total Return:   {metrics['total_return_pct']:>12.2f}%")
    print(f"Num Trades:     {metrics['num_trades']:>12}")
    print(f"Win Rate:       {metrics['win_rate_pct']:>12.2f}%")
    print(f"Avg Win:        ${metrics['avg_win']:>12,.2f}")
    print(f"Avg Loss:       ${metrics['avg_loss']:>12,.2f}")
    print(f"Profit Factor:  {metrics['profit_factor']:>12.2f}")
    print(f"Max Drawdown:   {metrics['max_drawdown_pct']:>12.2f}%")
    print(f"Sharpe Ratio:   {metrics['sharpe_ratio']:>12.2f}")
    print(f"Sortino Ratio:  {metrics['sortino_ratio']:>12.2f}")

    if not trades_df.empty:
        print()
        print(f"Trade Log (first 10)")
        print(f"{'-'*50}")
        display_cols = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "realized_pnl"]
        cols = [c for c in display_cols if c in trades_df.columns]
        print(trades_df[cols].head(10).to_string(index=False))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
