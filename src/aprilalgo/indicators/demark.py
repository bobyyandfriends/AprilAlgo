"""DeMark TD Sequential — Setup and Countdown with bull/bear signals.

Implements the core DeMark Sequential logic:
- **TD Setup**: 9 consecutive closes above/below the close 4 bars earlier
- **TD Countdown**: 13 bars where close relates to high/low 2 bars earlier
- Buy signals at completed bearish setups (exhaustion), sell at bullish setups
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def demark(df: pd.DataFrame, lookback: int = 4) -> pd.DataFrame:
    """Add DeMark TD Sequential columns and bull/bear signals to *df*.

    Columns added:
    - ``td_buy_setup`` — running count of bearish setup bars (1-9, 0 = reset)
    - ``td_sell_setup`` — running count of bullish setup bars (1-9, 0 = reset)
    - ``td_buy_countdown`` — running count toward buy countdown (1-13)
    - ``td_sell_countdown`` — running count toward sell countdown (1-13)
    - ``td_bull`` — True at completed buy signals (bearish exhaustion → reversal up)
    - ``td_bear`` — True at completed sell signals (bullish exhaustion → reversal down)
    """
    out = df.copy()
    n = len(out)
    close = out["close"].values

    buy_setup = np.zeros(n, dtype=int)
    sell_setup = np.zeros(n, dtype=int)
    buy_countdown = np.zeros(n, dtype=int)
    sell_countdown = np.zeros(n, dtype=int)
    td_bull = np.zeros(n, dtype=bool)
    td_bear = np.zeros(n, dtype=bool)

    buy_cd_active = False
    sell_cd_active = False
    buy_cd_count = 0
    sell_cd_count = 0

    for i in range(lookback, n):
        # --- TD Buy Setup: close < close[i - lookback] ---
        if close[i] < close[i - lookback]:
            buy_setup[i] = buy_setup[i - 1] + 1 if buy_setup[i - 1] < 9 else 9
        else:
            buy_setup[i] = 0

        # --- TD Sell Setup: close > close[i - lookback] ---
        if close[i] > close[i - lookback]:
            sell_setup[i] = sell_setup[i - 1] + 1 if sell_setup[i - 1] < 9 else 9
        else:
            sell_setup[i] = 0

        # Activate countdown on completed setup (count == 9)
        if buy_setup[i] == 9:
            buy_cd_active = True
            buy_cd_count = 0
        if sell_setup[i] == 9:
            sell_cd_active = True
            sell_cd_count = 0

        # --- TD Buy Countdown: close <= low[i - 2] ---
        if buy_cd_active and i >= 2:
            if close[i] <= out["low"].values[i - 2]:
                buy_cd_count += 1
            buy_countdown[i] = buy_cd_count
            if buy_cd_count >= 13:
                td_bull[i] = True
                buy_cd_active = False
                buy_cd_count = 0

        # --- TD Sell Countdown: close >= high[i - 2] ---
        if sell_cd_active and i >= 2:
            if close[i] >= out["high"].values[i - 2]:
                sell_cd_count += 1
            sell_countdown[i] = sell_cd_count
            if sell_cd_count >= 13:
                td_bear[i] = True
                sell_cd_active = False
                sell_cd_count = 0

        # A completed 9-count setup (without needing full countdown)
        # is also a weaker signal
        if buy_setup[i] == 9 and not td_bull[i]:
            td_bull[i] = True
        if sell_setup[i] == 9 and not td_bear[i]:
            td_bear[i] = True

    out["td_buy_setup"] = buy_setup
    out["td_sell_setup"] = sell_setup
    out["td_buy_countdown"] = buy_countdown
    out["td_sell_countdown"] = sell_countdown
    out["td_bull"] = td_bull
    out["td_bear"] = td_bear
    return out
