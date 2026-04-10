#%%
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

import src.utils.plot_tools as myplot
import src.data_processing.feature_engineering as get

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
# Seasonal Volume Prediction
# ---------------------------------------------------------------------------
#
# Player activity (and therefore Grand Exchange volume) cycles with the
# waking hours of the majority English-speaking player base: heavy trading
# during US/EU daytime, quiet overnight hours in NA.  The research plots
# (daytime_shade bands, totalvolume_time series) make this obvious visually.
#
# We exploit it with a pure linear seasonal model:
#
#   log_volume[t] = beta_0
#                 + sum_k a_k * sin(2*pi*k*h/24) + b_k * cos(2*pi*k*h/24)
#                 + sum_j c_j * sin(2*pi*j*d/7)  + d_j * cos(2*pi*j*d/7)
#                 + trend_coef * t_norm
#                 + eps
#
# Fit on hourly-resampled total market volume (sum across all items).
# A trend term catches the slow drift in activity over the sample.
# ---------------------------------------------------------------------------


def fourier_seasonal_features(
    index:           pd.DatetimeIndex,
    n_hourly:        int = 4,
    n_weekly:        int = 2,
    include_trend:   bool = True,
) -> pd.DataFrame:
    """Build sin/cos harmonics for daily + weekly cycles plus optional linear trend."""
    feats = {}
    # Hour-of-day (period = 24 hours)
    hours = index.hour + index.minute / 60.0
    for k in range(1, n_hourly + 1):
        feats[f'day_sin_{k}'] = np.sin(2 * np.pi * k * hours / 24)
        feats[f'day_cos_{k}'] = np.cos(2 * np.pi * k * hours / 24)

    # Day-of-week (period = 7 days) — with fractional hour for smoothness
    day_continuous = index.dayofweek + hours / 24
    for j in range(1, n_weekly + 1):
        feats[f'week_sin_{j}'] = np.sin(2 * np.pi * j * day_continuous / 7)
        feats[f'week_cos_{j}'] = np.cos(2 * np.pi * j * day_continuous / 7)

    if include_trend:
        feats['trend'] = np.linspace(0.0, 1.0, len(index))

    return pd.DataFrame(feats, index=index)


def fit_seasonal_volume_model(
    volume:          pd.Series,
    holdout_frac:    float = 0.2,
    n_hourly:        int = 4,
    n_weekly:        int = 2,
) -> dict:
    """
    Fit a linear seasonal model on log(volume) and return fit artefacts +
    in-sample / holdout predictions.
    """
    log_vol = np.log(volume.replace(0, np.nan).dropna())

    X_full = fourier_seasonal_features(log_vol.index, n_hourly=n_hourly, n_weekly=n_weekly)
    X_full = X_full.assign(intercept=1.0)

    n = len(log_vol)
    split = int(n * (1 - holdout_frac))

    X_train = X_full.iloc[:split].to_numpy()
    y_train = log_vol.iloc[:split].to_numpy()
    X_test  = X_full.iloc[split:].to_numpy()
    y_test  = log_vol.iloc[split:].to_numpy()

    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    y_train_pred = X_train @ coef
    y_test_pred  = X_test  @ coef

    # Pull predictions back to linear volume units
    train_pred_vol = pd.Series(np.exp(y_train_pred), index=log_vol.index[:split], name='pred')
    test_pred_vol  = pd.Series(np.exp(y_test_pred),  index=log_vol.index[split:], name='pred')
    train_actual   = pd.Series(np.exp(y_train),      index=log_vol.index[:split], name='actual')
    test_actual    = pd.Series(np.exp(y_test),       index=log_vol.index[split:], name='actual')

    metrics = {
        'train_mae':  mean_absolute_error(train_actual, train_pred_vol),
        'test_mae':   mean_absolute_error(test_actual,  test_pred_vol),
        'train_r2_log': r2_score(y_train, y_train_pred),
        'test_r2_log':  r2_score(y_test,  y_test_pred),
    }

    return {
        'coef':         coef,
        'feature_names': list(X_full.columns),
        'train_actual': train_actual,
        'train_pred':   train_pred_vol,
        'test_actual':  test_actual,
        'test_pred':    test_pred_vol,
        'metrics':      metrics,
    }


#%% ---- load data -----------------------------------------------------------
price_data = get.price_data
price_matrix_items, vol_matrix_items = get.item_data(price_data, wprice=True, datetime=True)

# Total market volume (sum across every tradable item) at 5-min resolution,
# then resample to hourly — matches the scale of the ACF plot in visual.py.
total_volume = vol_matrix_items.sum(axis=1)
total_volume_h = total_volume.resample('h').sum()
total_volume_h = total_volume_h[total_volume_h > 0]
print(f"Loaded {len(total_volume_h)} hourly observations of total market volume")

#%% ---- fit model -----------------------------------------------------------
result = fit_seasonal_volume_model(
    total_volume_h,
    holdout_frac=0.2,
    n_hourly=4,
    n_weekly=2,
)

print("\n===== Seasonal Volume Model =====")
for k, v in result['metrics'].items():
    print(f"  {k:<14s}: {v:,.4f}")

#%% ---- plot 1: actual vs predicted over the full series -------------------
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(result['train_actual'], color='white',    linewidth=0.6, label='Actual (train)')
ax1.plot(result['train_pred'],   color='#5DD4FF',  linewidth=0.8, label='Predicted (train)')
ax1.plot(result['test_actual'],  color='#FFD34F',  linewidth=0.6, label='Actual (test)')
ax1.plot(result['test_pred'],    color='red',      linewidth=0.8, label='Predicted (test)')
ax1.axvline(result['test_actual'].index[0], color='gray', linestyle='--', alpha=0.6)
ax1.set_title("Seasonal Volume Model — Full Series (log-scale)")
ax1.set_yscale('log')
ax1.set_ylabel("Hourly Volume")
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)
plt.xticks(rotation=45)
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "volume_seasonal_full.png", dpi=150, facecolor='#000000')

#%% ---- plot 2: zoomed 10-day slice showing the day/night cycle ------------
test_idx = result['test_actual'].index
zoom_end = test_idx[min(240, len(test_idx) - 1)]  # first 10 days of test
zoom_start = test_idx[0]

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(result['test_actual'].loc[zoom_start:zoom_end],
         color='white', linewidth=1.0, label='Actual')
ax2.plot(result['test_pred'].loc[zoom_start:zoom_end],
         color='#FF5D5D', linewidth=1.2, label='Predicted')
myplot.daytime_shade(result['test_actual'].loc[zoom_start:zoom_end], plot_type=ax2)
ax2.set_title("Seasonal Volume Model — 10-Day Hold-out Zoom")
ax2.set_ylabel("Hourly Volume")
ax2.legend()
ax2.grid(alpha=0.3)
plt.xticks(rotation=45)
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "volume_seasonal_zoom.png", dpi=150, facecolor='#000000')

#%% ---- plot 3: average volume profile by hour of day ----------------------
hourly_profile_actual = total_volume_h.groupby(total_volume_h.index.hour).mean()
predicted_all = pd.concat([result['train_pred'], result['test_pred']])
hourly_profile_pred = predicted_all.groupby(predicted_all.index.hour).mean()

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(hourly_profile_actual.index, hourly_profile_actual.values,
         marker='o', color='white', label='Actual (avg)')
ax3.plot(hourly_profile_pred.index, hourly_profile_pred.values,
         marker='o', color='#5DD4FF', label='Predicted (avg)')
ax3.axvspan(11, 20, color='gray', alpha=0.2, label='US daytime (EST)')
ax3.set_title("Mean Hourly Volume Profile by Hour-of-Day (US/Eastern)")
ax3.set_xlabel("Hour of Day")
ax3.set_ylabel("Mean Hourly Volume")
ax3.set_xticks(range(0, 24, 2))
ax3.legend()
ax3.grid(alpha=0.3)
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "volume_seasonal_hour_profile.png", dpi=150, facecolor='#000000')

#%% ---- plot 4: holdout residual diagnostics -------------------------------
residuals = result['test_actual'] - result['test_pred']
residuals_pct = residuals / result['test_actual'] * 100

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 7),
                                  gridspec_kw={'height_ratios': [2, 1]})
ax4a.plot(residuals_pct, color='#FFD34F', linewidth=0.6)
ax4a.axhline(0, color='white', linestyle='--', alpha=0.6)
ax4a.set_title("Holdout Residuals (% of actual)")
ax4a.set_ylabel("Residual (%)")
ax4a.grid(alpha=0.3)

ax4b.hist(residuals_pct.dropna(), bins=50, color='#5DD4FF',
          edgecolor='black', alpha=0.7)
ax4b.axvline(0, color='white', linestyle='--', alpha=0.6)
ax4b.set_title(
    f"Residual distribution  |  "
    f"mean={residuals_pct.mean():.2f}%  "
    f"std={residuals_pct.std():.2f}%"
)
ax4b.set_xlabel("Residual (%)")
ax4b.grid(alpha=0.3)

plt.setp(ax4a.get_xticklabels(), rotation=45)
fig4.tight_layout()
fig4.savefig(OUTPUT_DIR / "volume_seasonal_residuals.png", dpi=150, facecolor='#000000')

print(f"\nPlots saved to {OUTPUT_DIR}")
plt.show()
