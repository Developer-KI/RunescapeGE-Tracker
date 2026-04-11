from abc import ABC, abstractmethod

import pandas as pd

from src.backtester.engine import Signal, Portfolio


class Strategy(ABC):
    """
    Subclass this and implement `init` and `next`.

    `init` receives the full lookback window of data up to the current bar so
    you can pre-compute indicators, fit models, etc.

    `next` is called on every bar and must return a dict of
    {item_id: (Signal, quantity)} for each item the strategy wants to act on.
    Quantity must be a strictly positive integer — direction is expressed by
    the Signal (BUY/SELL), not by the sign of `quantity`. Orders with a
    non-positive quantity are ignored by the engine. Returning an empty dict
    or omitting an item means HOLD.
    """

    @abstractmethod
    def init(self, price_matrix: pd.DataFrame, vol_matrix: pd.DataFrame,
             high_matrix: pd.DataFrame | None, low_matrix: pd.DataFrame | None) -> None:
        ...

    @abstractmethod
    def next(self, idx: int, row: pd.Series, portfolio: Portfolio,
             price_matrix: pd.DataFrame, vol_matrix: pd.DataFrame) -> dict[int, tuple[Signal, int]]:
        """
        Parameters
        ----------
        idx : int
            Current integer position in the price matrix.
        row : pd.Series
            Current bar's weighted prices keyed by item_id.
        portfolio : Portfolio
            Live portfolio state.
        price_matrix, vol_matrix : pd.DataFrame
            Full matrices — the strategy may look back up to idx (inclusive).

        Returns
        -------
        dict mapping item_id -> (Signal, quantity), where `quantity` is a
        strictly positive integer.
        """
        ...
