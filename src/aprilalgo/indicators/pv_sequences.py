"""Price-Volume (PV) Sequences — state transition encoding.

Encodes each bar into one of four states based on whether price and volume
are increasing or decreasing vs the prior bar:
- PU_VU (Price Up, Volume Up) — strong conviction move up
- PU_VD (Price Up, Volume Down) — weak/fading move up
- PD_VU (Price Down, Volume Up) — strong conviction move down
- PD_VD (Price Down, Volume Down) — weak/fading move down

Crossovers between states and consecutive-state streaks generate signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PU_VU = 0
PU_VD = 1
PD_VU = 2
PD_VD = 3

_STATE_NAMES = {PU_VU: "PU_VU", PU_VD: "PU_VD", PD_VU: "PD_VU", PD_VD: "PD_VD"}


def pv_sequences(
    df: pd.DataFrame,
    streak_threshold: int = 3,
) -> pd.DataFrame:
    """Add Price-Volume state and bull/bear signals to *df*.

    Columns added:
    - ``pv_state`` — numeric state (0=PU_VU, 1=PU_VD, 2=PD_VU, 3=PD_VD)
    - ``pv_state_name`` — human-readable state label
    - ``pv_streak`` — consecutive bars in the same state
    - ``pv_bull`` — True when bullish pattern detected (PU_VU streak, or
      transition from PD to PU with volume expansion)
    - ``pv_bear`` — True when bearish pattern detected (PD_VU streak, or
      transition from PU to PD with volume expansion)
    """
    out = df.copy()
    n = len(out)

    price_up = (out["close"].values > np.roll(out["close"].values, 1))
    vol_up = (out["volume"].values > np.roll(out["volume"].values, 1))
    price_up[0] = False
    vol_up[0] = False

    states = np.where(
        price_up & vol_up, PU_VU,
        np.where(price_up & ~vol_up, PU_VD,
                 np.where(~price_up & vol_up, PD_VU, PD_VD))
    )

    streaks = np.ones(n, dtype=int)
    for i in range(1, n):
        if states[i] == states[i - 1]:
            streaks[i] = streaks[i - 1] + 1
        else:
            streaks[i] = 1

    prev_states = np.roll(states, 1)
    prev_states[0] = states[0]

    # Bullish: PU_VU streak OR transition from PD_VU to PU_VU (sellers exhausted)
    bull = (
        ((states == PU_VU) & (streaks >= streak_threshold))
        | ((states == PU_VU) & (prev_states == PD_VU))
    )

    # Bearish: PD_VU streak OR transition from PU_VU to PD_VU (buyers exhausted)
    bear = (
        ((states == PD_VU) & (streaks >= streak_threshold))
        | ((states == PD_VU) & (prev_states == PU_VU))
    )

    out["pv_state"] = states
    out["pv_state_name"] = [_STATE_NAMES[s] for s in states]
    out["pv_streak"] = streaks
    out["pv_bull"] = bull
    out["pv_bear"] = bear
    return out
