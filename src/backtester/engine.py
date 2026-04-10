#%%
import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtester.strategy import Strategy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.utils.model_tools as tools
import src.utils.plot_tools as myplot

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
# Data types
# ---------------------------------------------------------------------------

class Signal(Enum):
    BUY  = 1
    SELL = -1
    HOLD = 0


@dataclass
class Trade:
    item_id:     int
    side:        Signal          # BUY or SELL
    quantity:    int
    price:       float           # execution price (GP)
    timestamp:   pd.Timestamp
    cost:        float = 0.0     # total GP exchanged (negative = spent)


@dataclass
class Position:
    item_id:     int
    quantity:    int
    avg_entry:   float
    entry_time:  pd.Timestamp


@dataclass
class Portfolio:
    cash:        float
    positions:   dict = field(default_factory=dict)   # item_id -> Position
    trade_log:   list = field(default_factory=list)    # list[Trade]

    def equity(self, current_prices: pd.Series) -> float:
        holdings_value = sum(
            pos.quantity * current_prices.get(item_id, pos.avg_entry)
            for item_id, pos in self.positions.items()
        )
        return self.cash + holdings_value

    def holding(self, item_id: int) -> int:
        pos = self.positions.get(item_id)
        return pos.quantity if pos else 0


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

GE_TAX_RATE = 0.01  # 1% tax on sells

class Backtest:
    """
    Event-driven backtester that steps through a price matrix bar-by-bar.

    Parameters
    ----------
    strategy : Strategy
        A strategy instance implementing `init` and `next`.
    price_matrix : pd.DataFrame
        Weighted-price matrix (DatetimeIndex x item_id columns).
    vol_matrix : pd.DataFrame
        Volume matrix aligned to price_matrix.
    high_matrix, low_matrix : pd.DataFrame | None
        Optional high/low price matrices for spread-aware execution.
        When provided, buys execute at avgHighPrice, sells at avgLowPrice.
        When None, both sides execute at wprice (mid).
    initial_cash : float
        Starting GP balance.
    ge_tax : bool
        Apply the 1% Grand Exchange tax on sell proceeds.
    warmup : int
        Number of bars to skip before the strategy starts receiving signals
        (useful for indicator lookback).
    """

    def __init__(
        self,
        strategy:       "Strategy",
        price_matrix:   pd.DataFrame,
        vol_matrix:     pd.DataFrame,
        high_matrix:    pd.DataFrame | None = None,
        low_matrix:     pd.DataFrame | None = None,
        initial_cash:   float = 10_000_000,
        ge_tax:         bool = True,
        warmup:         int = 0,
    ):
        self.strategy     = strategy
        self.price_matrix = price_matrix
        self.vol_matrix   = vol_matrix
        self.high_matrix  = high_matrix
        self.low_matrix   = low_matrix
        self.initial_cash = initial_cash
        self.ge_tax       = ge_tax
        self.warmup       = warmup

        # results populated after run()
        self.equity_curve: pd.Series | None = None
        self.portfolio:    Portfolio | None  = None

    # ---- execution helpers ------------------------------------------------

    def _buy_price(self, item_id: int, idx: int) -> float:
        if self.high_matrix is not None:
            return self.high_matrix.iloc[idx][item_id]
        return self.price_matrix.iloc[idx][item_id]

    def _sell_price(self, item_id: int, idx: int) -> float:
        if self.low_matrix is not None:
            return self.low_matrix.iloc[idx][item_id]
        return self.price_matrix.iloc[idx][item_id]

    def _execute(self, signal: Signal, item_id: int, qty: int,
                 idx: int, timestamp: pd.Timestamp, portfolio: Portfolio) -> Trade | None:
        if qty <= 0:
            return None

        if signal == Signal.BUY:
            price = self._buy_price(item_id, idx)
            total_cost = price * qty
            if total_cost > portfolio.cash:
                qty = int(portfolio.cash // price)
                if qty <= 0:
                    return None
                total_cost = price * qty

            portfolio.cash -= total_cost

            if item_id in portfolio.positions:
                pos = portfolio.positions[item_id]
                new_qty = pos.quantity + qty
                pos.avg_entry = (pos.avg_entry * pos.quantity + price * qty) / new_qty
                pos.quantity = new_qty
            else:
                portfolio.positions[item_id] = Position(item_id, qty, price, timestamp)

            trade = Trade(item_id, Signal.BUY, qty, price, timestamp, cost=-total_cost)

        elif signal == Signal.SELL:
            held = portfolio.holding(item_id)
            if held <= 0:
                return None
            qty = min(qty, held)
            price = self._sell_price(item_id, idx)
            proceeds = price * qty
            if self.ge_tax:
                proceeds *= (1 - GE_TAX_RATE)

            portfolio.cash += proceeds
            pos = portfolio.positions[item_id]
            pos.quantity -= qty
            if pos.quantity <= 0:
                del portfolio.positions[item_id]

            trade = Trade(item_id, Signal.SELL, qty, price, timestamp, cost=proceeds)
        else:
            return None

        portfolio.trade_log.append(trade)
        return trade

    # ---- main loop --------------------------------------------------------

    def run(self) -> "BacktestResult":
        portfolio = Portfolio(cash=self.initial_cash)
        self.portfolio = portfolio

        self.strategy.init(self.price_matrix, self.vol_matrix,
                           self.high_matrix, self.low_matrix)

        equity_values = []
        timestamps = []
        n = len(self.price_matrix)

        for idx in range(n):
            row = self.price_matrix.iloc[idx]
            ts = self.price_matrix.index[idx]

            if idx >= self.warmup:
                signals = self.strategy.next(idx, row, portfolio,
                                             self.price_matrix, self.vol_matrix)
                for item_id, (sig, qty) in signals.items():
                    self._execute(sig, item_id, qty, idx, ts, portfolio)

            equity_values.append(portfolio.equity(row))
            timestamps.append(ts)

        self.equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(timestamps),
                                      name='equity')

        return BacktestResult(
            equity_curve=self.equity_curve,
            trade_log=portfolio.trade_log,
            initial_cash=self.initial_cash,
            price_matrix=self.price_matrix,
        )


# ---------------------------------------------------------------------------
# Results & metrics
# ---------------------------------------------------------------------------

class BacktestResult:
    def __init__(self, equity_curve: pd.Series, trade_log: list[Trade],
                 initial_cash: float, price_matrix: pd.DataFrame):
        self.equity_curve = equity_curve
        self.trade_log    = trade_log
        self.initial_cash = initial_cash
        self.price_matrix = price_matrix

    # ---- core metrics -----------------------------------------------------

    @property
    def total_return(self) -> float:
        return (self.equity_curve.iloc[-1] / self.initial_cash) - 1

    @property
    def returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()

    @property
    def sharpe(self, periods_per_day: int = 288) -> float:
        """Annualised Sharpe (assuming 5-min bars, 288 bars/day, 365 days)."""
        r = self.returns
        if r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(periods_per_day * 365))

    @property
    def sortino(self, periods_per_day: int = 288) -> float:
        r = self.returns
        downside = r[r < 0].std()
        if downside == 0:
            return 0.0
        return float(r.mean() / downside * np.sqrt(periods_per_day * 365))

    @property
    def max_drawdown(self) -> float:
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax
        return float(drawdown.min())

    @property
    def num_trades(self) -> int:
        return len(self.trade_log)

    @property
    def win_rate(self) -> float:
        """Win rate based on completed round-trip trades (buy then sell)."""
        buys: dict[int, list[Trade]] = {}
        wins = 0
        total = 0
        for t in self.trade_log:
            if t.side == Signal.BUY:
                buys.setdefault(t.item_id, []).append(t)
            elif t.side == Signal.SELL and t.item_id in buys and buys[t.item_id]:
                entry = buys[t.item_id].pop(0)
                total += 1
                if t.price > entry.price:
                    wins += 1
        return wins / total if total > 0 else 0.0

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame(columns=['timestamp', 'item_id', 'side', 'quantity', 'price', 'cost'])
        records = [{
            'timestamp': t.timestamp,
            'item_id':   t.item_id,
            'item_name': tools.item_name(t.item_id),
            'side':      t.side.name,
            'quantity':  t.quantity,
            'price':     t.price,
            'cost':      t.cost,
        } for t in self.trade_log]
        return pd.DataFrame(records)

    def summary(self) -> dict:
        return {
            'initial_cash':  self.initial_cash,
            'final_equity':  self.equity_curve.iloc[-1],
            'total_return':  f"{self.total_return:.4%}",
            'sharpe':        round(self.sharpe, 3),
            'sortino':       round(self.sortino, 3),
            'max_drawdown':  f"{self.max_drawdown:.4%}",
            'num_trades':    self.num_trades,
            'win_rate':      f"{self.win_rate:.2%}",
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n" + "=" * 50)
        print("  BACKTEST RESULTS")
        print("=" * 50)
        for k, v in s.items():
            print(f"  {k:<16s}: {v}")
        print("=" * 50 + "\n")

    # ---- plotting ---------------------------------------------------------

    def plot_equity(self, show: bool = False) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.equity_curve, color='cyan', linewidth=1)
        ax.axhline(self.initial_cash, color='gray', linestyle='--', alpha=0.6, label='Starting Cash')
        ax.set_title("Equity Curve")
        ax.set_ylabel("GP")
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid()
        plt.xticks(rotation=45)
        if show:
            plt.show()
        return fig, ax

    def plot_drawdown(self, show: bool = False) -> tuple[plt.Figure, plt.Axes]:
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax * 100

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.4)
        ax.plot(drawdown, color='red', linewidth=0.8)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Time")
        ax.grid()
        plt.xticks(rotation=45)
        if show:
            plt.show()
        return fig, ax

    def plot_trades(self, item_id: int, show: bool = False) -> tuple[plt.Figure, plt.Axes]:
        if item_id not in self.price_matrix.columns:
            raise ValueError(f"Item {item_id} not in price matrix")

        fig, ax = plt.subplots(figsize=(12, 5))
        prices = self.price_matrix[item_id]
        ax.plot(prices, color='white', linewidth=0.8, label=tools.item_name(item_id))
        myplot.daytime_shade(prices, plot_type=ax)

        buys  = [t for t in self.trade_log if t.item_id == item_id and t.side == Signal.BUY]
        sells = [t for t in self.trade_log if t.item_id == item_id and t.side == Signal.SELL]

        if buys:
            ax.scatter([t.timestamp for t in buys], [t.price for t in buys],
                       marker='^', color='lime', s=60, zorder=5, label='Buy')
        if sells:
            ax.scatter([t.timestamp for t in sells], [t.price for t in sells],
                       marker='v', color='red', s=60, zorder=5, label='Sell')

        ax.set_title(f"Trades: {tools.item_name(item_id)} ({item_id})")
        ax.set_ylabel("GP")
        ax.legend()
        ax.grid()
        plt.xticks(rotation=45)
        if show:
            plt.show()
        return fig, ax

    def plot(self, show: bool = False) -> None:
        """Combined equity + drawdown plot."""
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1])

        ax_eq = fig.add_subplot(gs[0, 0])
        ax_eq.plot(self.equity_curve, color='cyan', linewidth=1)
        ax_eq.axhline(self.initial_cash, color='gray', linestyle='--', alpha=0.6)
        ax_eq.set_title("Equity Curve")
        ax_eq.set_ylabel("GP")
        ax_eq.grid()
        myplot.daytime_shade(self.equity_curve, plot_type=ax_eq)
        plt.setp(ax_eq.get_xticklabels(), visible=False)

        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax * 100
        ax_dd = fig.add_subplot(gs[1, 0], sharex=ax_eq)
        ax_dd.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.4)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.grid()

        for label in ax_dd.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

        if show:
            plt.show()
