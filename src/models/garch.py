#%%
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult


def _prep_returns(data: pd.Series, scale: float = 100.0) -> pd.Series:
    """
    Convert a price series to log returns, dropping NaNs.

    GARCH fits are numerically better behaved when returns are scaled
    (the `arch` package itself warns when |y| is very small), so we scale
    by `scale` (default 100 → percent returns).
    """
    if not isinstance(data, pd.Series):
        raise ValueError("data must be a pandas Series of prices")

    prices = data.astype(float).dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns * scale


def GARCHFit(
    data:           pd.Series,
    holdout:        int = 0,
    p:              int = 1,
    q:              int = 1,
    mean:           str = "Constant",
    vol:            str = "GARCH",
    dist:           str = "normal",
    scale:          float = 100.0,
    is_returns:     bool = False,
) -> tuple[ARCHModelResult, pd.Series, float]:
    """
    Fit a GARCH(p, q) model to a price (or return) series.

    Parameters
    ----------
    data : pd.Series
        Price series by default. If `is_returns=True`, already-computed
        returns are passed through (but still scaled by `scale`).
    holdout : int
        Number of trailing observations reserved as a test set.
        The model is only fit on `data[:-holdout]`.
    p, q : int
        GARCH lag orders.
    mean : str
        Mean model: 'Constant', 'Zero', 'AR', 'HAR', etc.
    vol : str
        Volatility model: 'GARCH', 'EGARCH', 'GJR-GARCH' (via 'GARCH' with o>0), etc.
    dist : str
        Innovation distribution: 'normal', 't', 'skewt', 'ged'.
    scale : float
        Multiplicative scaling applied to returns before fitting
        (so the forecast output is in the same scaled units).
    is_returns : bool
        If True, treat `data` as a return series.

    Returns
    -------
    fitted : ARCHModelResult
        Fitted arch model object.
    train_returns : pd.Series
        Scaled returns used to fit the model.
    scale : float
        The scale factor (returned for downstream unscaling).
    """
    if is_returns:
        returns = data.astype(float).dropna() * scale
    else:
        returns = _prep_returns(data, scale=scale)

    if holdout > 0:
        train_returns = returns.iloc[:-holdout]
    else:
        train_returns = returns

    model = arch_model(
        train_returns,
        mean=mean,
        vol=vol,
        p=p,
        q=q,
        dist=dist,
    )
    fitted = model.fit(disp="off")
    return fitted, train_returns, scale


def GARCHForecast(
    fitted:     ARCHModelResult,
    horizon:    int = 1,
    scale:      float = 100.0,
    reindex:    bool = False,
) -> pd.DataFrame:
    """
    Point forecast of conditional mean and volatility for the next
    `horizon` steps.

    Returns
    -------
    pd.DataFrame with columns:
        - 'mean'         — forecasted return (unscaled back)
        - 'variance'     — forecasted conditional variance (unscaled back)
        - 'volatility'   — sqrt(variance), i.e. forecasted std-dev of returns
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    fc = fitted.forecast(horizon=horizon, reindex=reindex)

    # arch returns a DataFrame with one row per forecast origin; we want
    # the last row (the most recent origin) as a horizon-length vector.
    mean_row = fc.mean.iloc[-1].to_numpy()
    var_row  = fc.variance.iloc[-1].to_numpy()

    # Unscale: returns were multiplied by `scale` before fitting,
    # so variance was multiplied by scale**2.
    mean_unscaled = mean_row / scale
    var_unscaled  = var_row / (scale ** 2)
    vol_unscaled  = np.sqrt(var_unscaled)

    return pd.DataFrame(
        {
            "mean":       mean_unscaled,
            "variance":   var_unscaled,
            "volatility": vol_unscaled,
        },
        index=pd.RangeIndex(1, horizon + 1, name="h"),
    )


def GARCHForecastDensity(
    fitted:         ARCHModelResult,
    horizon:        int = 1,
    n_simulations:  int = 10_000,
    scale:          float = 100.0,
    last_price:     float | None = None,
    seed:           int | None = None,
) -> dict:
    """
    Monte-Carlo density forecast of the next `horizon` returns (and,
    optionally, prices) via simulation from the fitted GARCH process.

    Simulation uses the analytic forecast's conditional variance path and
    draws innovations from the model's fitted distribution to build a
    distribution of cumulative return paths.

    Parameters
    ----------
    fitted : ARCHModelResult
        Fitted GARCH model (from `GARCHFit`).
    horizon : int
        Number of steps to forecast ahead.
    n_simulations : int
        Number of Monte-Carlo paths.
    scale : float
        The same scale factor used at fitting time.
    last_price : float | None
        If provided, simulated returns are compounded onto this price to
        produce a distribution of future price paths.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        - 'return_paths'    : (n_simulations, horizon) simulated return draws (unscaled)
        - 'price_paths'     : (n_simulations, horizon) simulated prices (or None)
        - 'mean'            : horizon-length mean across sims
        - 'median'          : horizon-length median across sims
        - 'std'             : horizon-length std across sims
        - 'quantiles'       : dict of {q: horizon-length array}, q in (0.05, 0.25, 0.5, 0.75, 0.95)
        - 'terminal_summary': dict of summary stats at t+horizon
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")

    rng = np.random.default_rng(seed)

    # Use the analytic forecast for the conditional variance path and mean path.
    fc = fitted.forecast(horizon=horizon, reindex=False)
    mean_path_scaled = fc.mean.iloc[-1].to_numpy()        # shape (horizon,)
    var_path_scaled  = fc.variance.iloc[-1].to_numpy()    # shape (horizon,)
    std_path_scaled  = np.sqrt(var_path_scaled)

    # Draw standardised innovations from the fitted distribution if possible
    # (captures fat tails for 't', 'skewt', 'ged'); fall back to standard normal.
    try:
        dist_params = fitted.params[fitted.model.distribution.parameter_names()]
        innovations = fitted.model.distribution.simulate(dist_params.to_numpy())(
            size=(n_simulations, horizon)
        )
    except Exception:
        innovations = rng.standard_normal((n_simulations, horizon))

    # Simulated scaled returns: r_t = mu_t + sigma_t * z_t
    sim_returns_scaled = mean_path_scaled[None, :] + std_path_scaled[None, :] * innovations

    # Unscale back to raw log returns
    sim_returns = sim_returns_scaled / scale

    # Summary statistics per horizon step
    mean_per_step   = sim_returns.mean(axis=0)
    median_per_step = np.median(sim_returns, axis=0)
    std_per_step    = sim_returns.std(axis=0)

    q_levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    quantiles = {q: np.quantile(sim_returns, q, axis=0) for q in q_levels}

    # Optional price paths by compounding simulated log returns
    price_paths = None
    terminal_summary: dict = {}
    if last_price is not None:
        cum_log_returns = np.cumsum(sim_returns, axis=1)
        price_paths = last_price * np.exp(cum_log_returns)
        terminal_prices = price_paths[:, -1]
        terminal_summary = {
            "mean":     float(terminal_prices.mean()),
            "median":   float(np.median(terminal_prices)),
            "std":      float(terminal_prices.std()),
            "q05":      float(np.quantile(terminal_prices, 0.05)),
            "q25":      float(np.quantile(terminal_prices, 0.25)),
            "q75":      float(np.quantile(terminal_prices, 0.75)),
            "q95":      float(np.quantile(terminal_prices, 0.95)),
        }
    else:
        cum_returns = np.cumsum(sim_returns, axis=1)
        terminal_cum = cum_returns[:, -1]
        terminal_summary = {
            "mean":     float(terminal_cum.mean()),
            "median":   float(np.median(terminal_cum)),
            "std":      float(terminal_cum.std()),
            "q05":      float(np.quantile(terminal_cum, 0.05)),
            "q25":      float(np.quantile(terminal_cum, 0.25)),
            "q75":      float(np.quantile(terminal_cum, 0.75)),
            "q95":      float(np.quantile(terminal_cum, 0.95)),
        }

    return {
        "return_paths":     sim_returns,
        "price_paths":      price_paths,
        "mean":             mean_per_step,
        "median":           median_per_step,
        "std":              std_per_step,
        "quantiles":        quantiles,
        "terminal_summary": terminal_summary,
    }


def GARCH(
    data:           pd.Series,
    holdout:        int = 0,
    horizon:        int = 1,
    p:              int = 1,
    q:              int = 1,
    mean:           str = "Constant",
    vol:            str = "GARCH",
    dist:           str = "normal",
    scale:          float = 100.0,
    is_returns:     bool = False,
    density:        bool = False,
    n_simulations:  int = 10_000,
    seed:           int | None = None,
) -> tuple:
    """
    Convenience wrapper: fit + forecast (and optionally a density forecast).

    Returns
    -------
    (fitted_model, point_forecast_df, density_dict_or_None)
    """
    fitted, _, scale = GARCHFit(
        data=data, holdout=holdout, p=p, q=q,
        mean=mean, vol=vol, dist=dist, scale=scale, is_returns=is_returns,
    )

    point_forecast = GARCHForecast(fitted, horizon=horizon, scale=scale)

    density_forecast = None
    if density:
        last_price = None
        if not is_returns:
            train_prices = data.astype(float).dropna()
            if holdout > 0:
                train_prices = train_prices.iloc[:-holdout]
            last_price = float(train_prices.iloc[-1])

        density_forecast = GARCHForecastDensity(
            fitted,
            horizon=horizon,
            n_simulations=n_simulations,
            scale=scale,
            last_price=last_price,
            seed=seed,
        )

    return fitted, point_forecast, density_forecast
