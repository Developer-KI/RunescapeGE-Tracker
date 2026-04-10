#%%
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import src.utils.model_tools as tools
import src.utils.plot_tools as myplot
import src.data_processing.feature_engineering as get
from src.backtester.engine import Signal, Backtest, Portfolio
from src.backtester.strategy import Strategy

plt.rcParams.update({
    "figure.facecolor": "#000000",
    "axes.facecolor": "#000000",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "gray",
    "legend.facecolor": "#303030",
    "legend.labelcolor": "#A1A1A1",
    "axes.titlecolor": "#A1A1A1"
})

# ---------------------------------------------------------------------------
# Cointegrated Group Mean-Reversion Strategy
# ---------------------------------------------------------------------------
#
# Given a group of items with a pre-computed Johansen cointegrating vector,
# the strategy tracks the stationary spread:
#
#   spread[t] = prices[t] @ eigenvector
#
# A rolling z-score of the spread flags dislocations from the long-run
# equilibrium.  Because OSRS has no shorting mechanic, we only trade the
# long leg: when the z-score is sufficiently negative, the positive-weighted
# basket of items is undervalued relative to the equilibrium — we buy it,
# then unwind when the spread reverts past a neutral level.
#
# Capital per leg is allocated in proportion to the (normalised) positive
# weights of the eigenvector so that the basket as a whole mirrors the
# cointegrating combination.
# ---------------------------------------------------------------------------


class CointegratedGroupReversion(Strategy):
    """
    Mean-reversion strategy over a group of cointegrated items.

    Parameters
    ----------
    items : list[int]
        Item ids in the group, in the same order as `coint_vector`.
    coint_vector : np.ndarray
        Johansen cointegrating eigenvector of length `len(items)`.
    z_window : int
        Rolling window (in bars) for the spread z-score.
    entry_z : float
        Z-score threshold for entering a long-basket position (spread low).
    exit_z : float
        Z-score threshold to exit an open position (mean reversion).
    stop_z : float
        Hard stop-loss z-score (spread moves further against us).
    capital_pct : float
        Fraction of portfolio equity to deploy when entering a trade.
    """

    def __init__(
        self,
        items:         list[int],
        coint_vector:  np.ndarray,
        z_window:      int   = 288,     # 1 day of 5-min bars
        entry_z:       float = -2.0,
        exit_z:        float = -0.2,
        stop_z:        float = -3.5,
        capital_pct:   float = 0.9,
    ):
        if len(items) != len(coint_vector):
            raise ValueError("items and coint_vector must have the same length")

        self.items        = list(items)
        self.coint_vector = np.asarray(coint_vector, dtype=float)
        self.z_window     = z_window
        self.entry_z      = entry_z
        self.exit_z       = exit_z
        self.stop_z       = stop_z
        self.capital_pct  = capital_pct

        # Normalised positive weights used for capital allocation on entry.
        # If the leading eigenvector happens to have mostly negative loadings,
        # we flip its sign so the positive-weighted basket is non-empty.
        vec = self.coint_vector.copy()
        if np.sum(vec > 0) == 0:
            vec = -vec
            self.coint_vector = vec
        pos_mask = vec > 0
        pos_weights = np.where(pos_mask, vec, 0.0)
        total = pos_weights.sum()
        self._alloc_weights = pos_weights / total if total > 0 else pos_weights

        # Populated in init()
        self._spread:      pd.Series | None = None
        self._spread_mean: pd.Series | None = None
        self._spread_std:  pd.Series | None = None
        self._in_position = False

    # ---- Strategy interface -----------------------------------------------

    def init(self, price_matrix: pd.DataFrame, vol_matrix: pd.DataFrame,
             high_matrix: pd.DataFrame | None, low_matrix: pd.DataFrame | None) -> None:

        missing = [i for i in self.items if i not in price_matrix.columns]
        if missing:
            raise ValueError(f"Items missing from price_matrix: {missing}")

        group_prices = price_matrix[self.items].astype(float)
        spread = group_prices.values @ self.coint_vector
        self._spread = pd.Series(spread, index=price_matrix.index, name='spread')

        self._spread_mean = self._spread.rolling(self.z_window).mean()
        self._spread_std  = self._spread.rolling(self.z_window).std()

    def next(self, idx: int, row: pd.Series, portfolio: Portfolio,
             price_matrix: pd.DataFrame, vol_matrix: pd.DataFrame) -> dict[int, tuple[Signal, int]]:

        assert self._spread is not None and self._spread_mean is not None and self._spread_std is not None

        if idx < self.z_window:
            return {}

        mean = self._spread_mean.iloc[idx]
        std  = self._spread_std.iloc[idx]
        if std == 0 or np.isnan(std) or np.isnan(mean):
            return {}

        z = (self._spread.iloc[idx] - mean) / std
        orders: dict[int, tuple[Signal, int]] = {}

        # ---- position management --------------------------------------
        if self._in_position:
            # Hard stop: spread dislocated further against us
            if z < self.stop_z:
                for item_id in self.items:
                    held = portfolio.holding(item_id)
                    if held > 0:
                        orders[item_id] = (Signal.SELL, held)
                self._in_position = False
                return orders

            # Mean reversion take-profit
            if z > self.exit_z:
                for item_id in self.items:
                    held = portfolio.holding(item_id)
                    if held > 0:
                        orders[item_id] = (Signal.SELL, held)
                self._in_position = False
                return orders

            return {}

        # ---- entry: spread is well below equilibrium ------------------
        if z < self.entry_z:
            equity = portfolio.equity(row)
            budget = equity * self.capital_pct

            for item_id, w in zip(self.items, self._alloc_weights):
                if w <= 0:
                    continue
                price = row[item_id]
                if price <= 0 or np.isnan(price):
                    continue
                leg_budget = budget * w
                qty = int(leg_budget // price)
                if qty > 0:
                    orders[item_id] = (Signal.BUY, qty)

            if orders:
                self._in_position = True

        return orders


# ---------------------------------------------------------------------------
# Helper: fit Johansen on a group and return the leading eigenvector
# ---------------------------------------------------------------------------

def johansen_eigenvector(
    price_matrix: pd.DataFrame,
    items:        list[int],
    det_order:    int = 0,
    k_ar_diff:    int = 1,
    resample:     str | None = 'h',
) -> tuple[list[int], np.ndarray, dict]:
    """
    Fit the Johansen test on `items` and return the leading cointegrating
    eigenvector.  Items with no price coverage are dropped.

    Returns
    -------
    available_items : list[int]
        Items actually used (in column order of the eigenvector).
    leading_vec : np.ndarray
        The first cointegrating eigenvector.
    info : dict
        Diagnostic info: trace stat, critical values, n_coint at 95%.
    """
    available = [i for i in items if i in price_matrix.columns]
    if len(available) < 2:
        raise ValueError("Need at least 2 items with price coverage for Johansen test")

    group = price_matrix[available].astype(float)
    if resample is not None:
        group = group.resample(resample).mean()
    group = group.dropna()

    result = coint_johansen(group, det_order, k_ar_diff)

    n_vars = group.shape[1]
    n_coint_95 = int(sum(result.lr1[i] > result.cvt[i, 1] for i in range(n_vars)))

    info = {
        'trace_stats':   result.lr1,
        'crit_values':   result.cvt,
        'eigenvalues':   result.eig,
        'n_coint_95':    n_coint_95,
    }
    return available, result.evec[:, 0], info


# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- load data --------------------------------------------------------
    # Barrows pieces are low-volume items not included in the general
    # processed_data.csv, so we use the dedicated boss dataset
    # (mirrors research/cointegration.py).
    _, price_matrix, vol_matrix = get.boss_data(datetime=True)

    # ---- build the cointegrated group ------------------------------------
    # Use the Dharok Barrows set - known to be cointegrated since the
    # pieces only drop together from Dharok the Wretched.
    group_items = [
        tools.item_name("Dharok's helm"),
        tools.item_name("Dharok's platebody"),
        tools.item_name("Dharok's platelegs"),
        tools.item_name("Dharok's greataxe"),
    ]

    # Fit Johansen on hourly-resampled prices to get the cointegrating vector.
    items_used, coint_vec, info = johansen_eigenvector(
        price_matrix, group_items, det_order=0, k_ar_diff=1, resample='h'
    )
    print(f"\nJohansen test on Dharok set: {len(items_used)} items, "
          f"{info['n_coint_95']} cointegrating relations at 95%")
    print(f"Leading eigenvector: {coint_vec}")

    # ---- configure strategy -----------------------------------------------
    strategy = CointegratedGroupReversion(
        items        = items_used,
        coint_vector = coint_vec,
        z_window     = 288,     # 1-day rolling z-score
        entry_z      = -2.0,
        exit_z       = -0.2,
        stop_z       = -3.5,
        capital_pct  = 0.9,
    )

    # ---- run backtester ---------------------------------------------------
    bt = Backtest(
        strategy     = strategy,
        price_matrix = price_matrix,
        vol_matrix   = vol_matrix,
        initial_cash = 10_000_000,
        ge_tax       = True,
        warmup       = 288,    # warmup equal to z-score window
    )
    result = bt.run()

    # ---- results ----------------------------------------------------------
    OUTPUT_DIR = Path(__file__).resolve().parent / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    result.print_summary()

    # Equity curve + drawdown
    result.plot(show=False)
    plt.gcf().savefig(OUTPUT_DIR / "pairs_reversion_equity_old.png",
                      dpi=150, facecolor='#000000')
    plt.close(plt.gcf())

    # Trades on the first leg
    fig_trades, _ = result.plot_trades(items_used[0], show=False)
    fig_trades.savefig(OUTPUT_DIR / "pairs_reversion_trades_old.png",
                       dpi=150, facecolor='#000000')
    plt.close(fig_trades)

    # Spread z-score with entry/exit bands
    spread = strategy._spread
    spread_mean = strategy._spread_mean
    spread_std  = strategy._spread_std
    z = ((spread - spread_mean) / spread_std).dropna()

    fig_z, ax = plt.subplots(figsize=(12, 5))
    ax.plot(z, color='cyan', linewidth=0.8, label='Spread Z-score')
    ax.axhline(strategy.entry_z, color='lime',   linestyle='--', alpha=0.6, label='Entry')
    ax.axhline(strategy.exit_z,  color='yellow', linestyle='--', alpha=0.6, label='Exit')
    ax.axhline(strategy.stop_z,  color='red',    linestyle='--', alpha=0.6, label='Stop')
    ax.axhline(0, color='white', linestyle='-', alpha=0.3)
    ax.set_title("Dharok Group Spread Z-Score")
    ax.set_ylabel("z")
    ax.grid()
    ax.legend()
    plt.xticks(rotation=45)
    fig_z.savefig(OUTPUT_DIR / "pairs_reversion_spread_zscore_old.png",
                  dpi=150, facecolor='#000000', bbox_inches='tight')
    plt.close(fig_z)

    print(f"\nPlots saved to {OUTPUT_DIR}")