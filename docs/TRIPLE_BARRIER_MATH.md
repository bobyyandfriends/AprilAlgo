# Triple-barrier labeling — math and conventions

This matches `src/aprilalgo/labels/triple_barrier.py` and ARCHITECTURE.md §4.6 (profit target, stop loss, time limit).

## Setup

For a decision (row) index **i** with ascending time-ordered OHLC:

- **Entry price** \(p^{\mathrm{entry}}_i = C_i\) (close at the decision bar).
- **Upper barrier** (take-profit for a long view): \(U_i = p^{\mathrm{entry}}_i \cdot (1 + \rho_{\mathrm{up}})\).
- **Lower barrier** (stop-loss): \(L_i = p^{\mathrm{entry}}_i \cdot (1 - \rho_{\mathrm{dn}})\).
- **Vertical barrier**: at most **V** future bars are observed after **i** (bars \(i+1, \ldots, i+V\)). On **information-driven** bar series, each step is one aggregated bar (tick/volume/dollar), not necessarily a fixed calendar interval.

Bar **i** itself is not used to test touches; only **strictly future** highs and lows matter for path-dependent outcomes, so features aligned to **i** do not embed post-entry path into the entry definition.

## Per-bar checks

For each future bar \(j \in \{i+1, \ldots, i+V\}\):

- **Upper touched** if \(H_j \ge U_i\).
- **Lower touched** if the low price at \(j\) is at or below the lower barrier \(L^{\mathrm{bar}}_i = p^{\mathrm{entry}}_i \cdot (1 - \rho_{\mathrm{dn}})\), i.e. \(\mathrm{low}_j \le L^{\mathrm{bar}}_i\).

Scan **forward in time**. The **first** bar on which at least one barrier condition holds determines the outcome, except when both hold on the **same** bar (see below).

## Outcomes

- If the **lower** condition triggers first (alone, or under the same-bar policy): label **stop-loss** → `LABEL_STOP_LOSS` (-1).
- If the **upper** condition triggers first (alone): label **take-profit** → `LABEL_TAKE_PROFIT` (+1).
- If neither triggers through bar \(i+V\): **vertical exit** → `LABEL_VERTICAL_TIMEOUT` (0).

## Same-bar ambiguity

If **both** \(H_j \ge U_i\) and \(L_j \le L_i\) on the same bar \(j\), the intrabar path is unknown. The implementation uses an explicit policy:

- **`stop_loss_first`** (default): count the lower barrier as hit first (conservative for a long-horizon profit/stop interpretation).
- **`take_profit_first`**: count the upper barrier first.

Document which policy you use in experiments; it affects class balance slightly.

## Insufficient horizon

If \(i + V\) exceeds the last index of the series, the vertical barrier cannot be evaluated to completion. Those rows receive **NaN** labels (and NaN offsets).

## No “peeking”

- **Label at i** depends only on \(C_i\) (entry) and \(\{(H_j, L_j) : j = i+1, \ldots, i+V\}\) within the dataframe slice.
- Changing OHLC at indices **after** the first barrier hit does not change the label at **i** (see tests).

Purged k-fold and embargo for training are handled separately (see `DATA_SCHEMA.md` and future `cv.py`).
