#%%
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(name: str, fig=None) -> None:
    """Save the given figure (or the current one) to OUTPUT_DIR as PNG."""
    import matplotlib.pyplot as _plt
    if fig is None:
        fig = _plt.gcf()
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150,
                facecolor='#000000', bbox_inches='tight')

from src.models import exp_smoothing as myEXPS, hmm as myHMM, rfts as myRFTS
#%%
import  pandas as pd, numpy as np
from    src.utils.model_tools import (
    target_rolling_features, 
    create_feature_lags, 
    volatility_market, 
    item_name, 
    create_item_index
)
import  src.utils.plot_tools as myplot
from    src.models import (
    xgb as myXGB
    )
from    matplotlib import pyplot as plt
from    matplotlib import ticker as mticker
import  src.data_processing.feature_engineering as get
import src.data_processing.outlier_detection as outlier
from src.utils.model_tools import item_name
import seaborn as sns
#%%
price_data = get.price_data
price_matrix_items, vol_matrix_items = get.item_data(price_data, True)
#%%
update_dates, updates_hours_since, updates_days_since, _ = get.updates_announcements(price_matrix_items)
_, _, _, _, _, day_trig = get.cyclical_time(price_matrix_items)
equal_index, vprice_index = get.market_indices(price_matrix_items, vol_matrix_items)
_, boss_matrix_items, boss_vol_matrix_items = get.boss_data()
#%% 
random_item =  np.random.choice(price_matrix_items.columns)
target_item = random_item
target_item = 1603
smoothing_adjustment = 20

vol_items_reg = vol_matrix_items[[target_item]]
price_items_reg = price_matrix_items[[target_item]]
price_items_reg.columns = [f'{target_item}']
vol_items_reg.columns = [f'{target_item}_vol']
volatility_index = volatility_market(price_matrix_items, smoothing=smoothing_adjustment) 
reg_data = pd.concat([price_items_reg, vol_items_reg, volatility_index], axis=1).dropna()

df_time = create_feature_lags(reg_data, f'{target_item}', [1,6,8,10,12,30,288]) #hour, day
df_rolling = target_rolling_features(df_time, f'{target_item}', smoothing_adjustment)
master = df_rolling
master[['day sin','day cos']] = day_trig.loc[master.index]
master['hours since update']= updates_hours_since.loc[master.index]
master['days since update']= updates_days_since.loc[master.index]
#%% Run a RFTS 
holdout= round(price_matrix_items.shape[0]*0.2)

rfts_model, holdout_pred_rfts, train_outliers_rfts, holdout_outliers_rfts, cv_mae_rfts, cv_mase_rfts, train_cv_mae_rfts, train_cv_mase_rfts = myRFTS.RFTS(
    master, 
    target_col=         f'{target_item}', 
    outlier_window=     200, 
    detection =        'ewm2', 
    ewm_bounds=         [3,3], 
    mase_m=             1,
    holdout=            holdout, 
    outlier_threshold=  None, 
    max_depth=          5, 
    n_estimators=       400, 
    min_samples_leaf=   20
    )

predictions_rfts = rfts_model.predict(master.drop(f'{target_item}', axis=1))
myplot.plot_pred_vs_price(master, predictions=predictions_rfts, holdout_pred_n=holdout, lookback=5000, fill_outliers=train_outliers_rfts+holdout_outliers_rfts)
save_fig(f"rfts_pred_vs_price_{target_item}")
RFTS_residuals  = myplot.plot_residuals(master, rfts_model)
save_fig(f"rfts_residuals_{target_item}")
feature_importances = pd.Series(rfts_model.feature_importances_, index=master.columns[1:]).sort_values(ascending=False)
print(feature_importances)
#%% Run a XBG model TODO copy over RFTS changes
xgb_model, holdout_pred_xgb, train_outliers_xgb, holdout_outliers_xgb, cv_mae_xgb, cv_mase_xgb, train_cv_mae_xgb, train_cv_mase_xgb = myXGB.XGB(
    master, 
    target_col=         f'{target_item}', 
    outlier_window=     200, 
    detection=          'ewm2', 
    ewm_bounds=         [0.07,0.07], 
    mase_m=             1,
    holdout=            holdout, 
    outlier_threshold=  None, 
    max_depth=          5, 
    n_estimators=       400, 
    time_splits=        5,
    learning_rate=      0.03,
    subsample=          0.4,
    colsample_bytree=   0.8,
    min_child_weight=   5,
    )
predictions_xgb = xgb_model.predict(master.drop(f'{target_item}', axis=1))
myplot.plot_pred_vs_price(master, predictions=predictions_xgb, holdout_pred_n=holdout, lookback=5000, fill_outliers=train_outliers_xgb+holdout_outliers_xgb)
save_fig(f"xgb_pred_vs_price_{target_item}")
feature_importances = pd.Series(xgb_model.feature_importances_, index=master.columns[1:]).sort_values(ascending=False)
print(feature_importances)
#%% Run an EXPS model
exps_holdout = 200
exps_training_size = master.shape[0] - exps_holdout
EXPSmodel, pred_exps = myEXPS.EXPSmooth(master.copy(), holdout=exps_training_size, predictions=exps_holdout)
actual_holdout_exps = master.iloc[-exps_holdout:, 0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(exps_holdout), actual_holdout_exps.values, label='Actual', color='white')
ax.plot(range(exps_holdout), pred_exps.values, label='EXPS Forecast', color='red')
ax.legend()
ax.set_title(f'{item_name(target_item)} [{target_item}] Exponential Smoothing: Forecast vs Actual')
ax.grid()
save_fig(f"exps_forecast_{target_item}", fig=fig)
plt.show()
#%% Run the optimized RFTS
optim, optimparam,best_test_idx = myRFTS.RFTSOptim(master,target_col=f'{target_item}',n_trials=5, holdout=500)
#%%
predictions_optim = optim.predict(master.drop(f'{target_item}', axis=1))
myplot.plot_pred_vs_price(master, predictions=predictions_optim, holdout_pred_n=holdout, lookback=0)
save_fig(f"rfts_optim_pred_vs_price_{target_item}")
#%%
myplot.test_train_error(master, param='max_depth',
                       exclude_param={},
                       model_class=myRFTS.RandomForestRegressor,
                       param_range=(1, 10))
save_fig(f"rfts_test_train_error_max_depth_{target_item}")
# %%
plt.figure(figsize=(10,5))
plt.plot(range(len(cv_mae_rfts)),cv_mae_rfts, label='Test')
plt.plot(range(len(train_cv_mae_rfts)),train_cv_mae_rfts, label='Train')
plt.legend()
save_fig(f"rfts_cv_mae_{target_item}")
plt.show()
# %%
plt.figure(figsize=(10,5))
plt.plot(range(len(cv_mae_xgb)),cv_mae_xgb, label='Test')
plt.plot(range(len(train_cv_mae_xgb)),train_cv_mae_xgb, label='Train')
plt.legend()
save_fig(f"xgb_cv_mae_{target_item}")
plt.show()
# %% ------------------------------------------------
myplot.plot_item_market_divergence(price_matrix_items, 1603, equal_index, '10min')
save_fig("item_market_divergence_1603")
# %% -------------------------------------------------
full_y = price_matrix_items[1603]
volume = vol_matrix_items[1603]
window=100
threshold = 1

fig, ax = myplot.plot_features(full_y, start='2025-08-4', end='2025-08-11')
save_fig("benchmark_full_y_slice", fig=fig)
log_return = np.log(full_y / full_y.shift(1))

ratio = np.abs(log_return) / volume

rolling_mean_ratio = ratio.rolling(window=window).mean()
rolling_std_ratio = ratio.rolling(window=window).std()

outlier_threshold = rolling_mean_ratio + rolling_std_ratio * threshold

# %%
fig, ax = myplot.plot_features(ratio, start='2025-08-4',end='2025-08-11')
ax.plot(outlier_threshold.loc['2025-08-4':'2025-08-11'], label='threshold')
fig.legend()
save_fig("benchmark_ratio_outlier_threshold", fig=fig)

# %% ============================================================
#    HMM Regime Classification Benchmark
#    ============================================================
#
# Fits a 3-state CategoricalHMM on rolling-threshold-classified
# observations (bearish / neutral / bullish) for the target item,
# then visualises the inferred hidden regimes on top of the price
# along with transition and emission diagnostics.
# --------------------------------------------------------------

# Build a single-column DataFrame keyed by a string column name so
# hmm_data_prep (which does `classified[item]`) can look it up.
hmm_item_col = f'{target_item}'
hmm_features = price_matrix_items[[target_item]].copy()
hmm_features.columns = [hmm_item_col]

# Call the underlying primitives directly instead of `item_threshold_hmm`
# so we never depend on the wrapper's embedded `plt.show()` for figure
# capture — every figure below is created and saved from its own handle.
n_symbols, n_samples, hmm_observations = myHMM.hmm_data_prep(
    hmm_features, hmm_item_col, window=100, diff_percent=0.1
)
hidden_states, log_likelihood, n_parameters, hmm_model = myHMM.hmm_train_score(
    hmm_observations, n_components=3, n_symbols=n_symbols, n_iter=1000
)
aic = 2 * n_parameters - 2 * log_likelihood
bic = n_parameters * np.log(n_samples) - 2 * log_likelihood
print(f"HMM [AIC, BIC] for {item_name(target_item)} [{target_item}]: [{aic:.2f}, {bic:.2f}]")

n_states = hmm_model.n_components

# ---- state-shaded price plot (our own, so we control the figure) ----
state_colors = {0: "red", 1: "gray", 2: "green"}
fig_hs, ax_hs = plt.subplots(figsize=(12, 5))
price_series = hmm_features[hmm_item_col]
timescale = price_series.index
for t in range(len(timescale) - 1):
    ax_hs.axvspan(timescale[t], timescale[t + 1],
                  color=state_colors.get(int(hidden_states[t]), 'gray'),
                  alpha=0.07)
ax_hs.plot(timescale, price_series.values, color='white', linewidth=0.8)
ax_hs.set_title(f"HMM Regime Classification ({item_name(target_item)})")
ax_hs.set_ylabel("Price (GP)")
ax_hs.set_xlabel("Time")
ax_hs.grid(alpha=0.3)
plt.setp(ax_hs.get_xticklabels(), rotation=45)
save_fig(f"hmm_states_vs_price_{target_item}", fig=fig_hs)
plt.show()

# ---- transition matrix heatmap ----
fig_tm, ax_tm = plt.subplots(figsize=(6, 5))
sns.heatmap(hmm_model.transmat_, annot=True, fmt=".3f",
            cmap='viridis', ax=ax_tm,
            xticklabels=[f'S{i}' for i in range(n_states)],
            yticklabels=[f'S{i}' for i in range(n_states)])
ax_tm.set_title(f"HMM Transition Matrix ({item_name(target_item)})")
ax_tm.set_xlabel("To state")
ax_tm.set_ylabel("From state")
save_fig(f"hmm_transition_matrix_{target_item}", fig=fig_tm)
plt.show()

# ---- emission matrix heatmap ----
# Only label the columns for symbols the model actually learnt.
symbol_labels = ['down', 'flat', 'up']
n_emit = hmm_model.emissionprob_.shape[1]
xticklabels = symbol_labels[:n_emit] if n_emit <= 3 else [f'sym{i}' for i in range(n_emit)]

fig_em, ax_em = plt.subplots(figsize=(6, 5))
sns.heatmap(hmm_model.emissionprob_, annot=True, fmt=".3f",
            cmap='magma', ax=ax_em,
            xticklabels=xticklabels,
            yticklabels=[f'S{i}' for i in range(n_states)])
ax_em.set_title(f"HMM Emission Probabilities ({item_name(target_item)})")
ax_em.set_xlabel("Observation symbol")
ax_em.set_ylabel("Hidden state")
save_fig(f"hmm_emission_matrix_{target_item}", fig=fig_em)
plt.show()

# ---- state occupancy bar chart ----
# Re-index the counts so every possible state shows a bar even if unused.
state_counts = (pd.Series(hidden_states)
                  .value_counts()
                  .reindex(range(n_states), fill_value=0)
                  .sort_index())
state_frac = state_counts / state_counts.sum()

bar_colors = ['#FF5D5D', '#AAAAAA', '#5DD45D']
fig_sd, ax_sd = plt.subplots(figsize=(8, 4))
ax_sd.bar([f'S{i}' for i in state_frac.index],
          state_frac.values,
          color=bar_colors[:n_states],
          edgecolor='white')
ax_sd.set_title(f"HMM State Occupancy ({item_name(target_item)})")
ax_sd.set_xlabel("Hidden state")
ax_sd.set_ylabel("Fraction of time")
ax_sd.grid(axis='y', alpha=0.3)
save_fig(f"hmm_state_occupancy_{target_item}", fig=fig_sd)
plt.show()

# %% ============================================================
#    GARCH Volatility Benchmark
#    ============================================================
#
# Fits a GARCH(1,1) with Student's t innovations (motivated by the
# fat tails seen in research/visual.py) on the target item's log
# returns, then produces:
#   1. fitted conditional volatility overlaid on absolute returns
#   2. standardised residual diagnostics (normality)
#   3. multi-step point forecast of conditional volatility
#   4. Monte-Carlo density forecast fan chart over prices
# --------------------------------------------------------------

from src.models.garch import GARCHFit, GARCHForecast, GARCHForecastDensity

garch_prices = price_matrix_items[target_item].astype(float).dropna()
# Subsample to make the fit snappier on the huge 5-min series
garch_prices = garch_prices.iloc[-8000:]

garch_holdout = 288  # last day reserved for forecast comparison
garch_fit, train_returns_scaled, garch_scale = GARCHFit(
    data       = garch_prices,
    holdout    = garch_holdout,
    p          = 1,
    q          = 1,
    mean       = "Constant",
    vol        = "GARCH",
    dist       = "t",
    scale      = 100.0,
)
print(garch_fit.summary())

# ---- plot 1: conditional volatility vs realised abs returns ----
cond_vol_scaled = garch_fit.conditional_volatility
cond_vol = cond_vol_scaled / garch_scale     # back to raw log-return units
train_returns = train_returns_scaled / garch_scale

fig_gv, ax_gv = plt.subplots(figsize=(12, 5))
ax_gv.plot(train_returns.index, train_returns.abs(),
           color='#AAAAAA', linewidth=0.5, label='|log return|')
ax_gv.plot(cond_vol.index, cond_vol,
           color='#FFD34F', linewidth=1.2, label='GARCH conditional σ')
ax_gv.set_title(f"GARCH(1,1)-t Conditional Volatility — {item_name(target_item)} [{target_item}]")
ax_gv.set_ylabel("Volatility / |return|")
ax_gv.legend()
ax_gv.grid(alpha=0.3)
plt.xticks(rotation=45)
save_fig(f"garch_conditional_volatility_{target_item}", fig=fig_gv)
plt.show()

# ---- plot 2: standardised residuals ----
std_resid = garch_fit.std_resid.dropna()

fig_sr, (ax_sr1, ax_sr2) = plt.subplots(1, 2, figsize=(12, 4))
ax_sr1.plot(std_resid.index, std_resid.values, color='#5DD4FF', linewidth=0.5)
ax_sr1.axhline(0, color='white', linestyle='--', alpha=0.5)
ax_sr1.set_title("Standardised residuals")
ax_sr1.grid(alpha=0.3)
plt.setp(ax_sr1.get_xticklabels(), rotation=45)

ax_sr2.hist(std_resid.values, bins=50, color='#5DD4FF',
            edgecolor='black', alpha=0.7, density=True)
ax_sr2.axvline(0, color='white', linestyle='--', alpha=0.5)
ax_sr2.set_title(
    f"Residual distribution  |  "
    f"skew={std_resid.skew():.2f}  "
    f"kurt={std_resid.kurtosis():.2f}"
)
ax_sr2.grid(alpha=0.3)
plt.tight_layout()
save_fig(f"garch_std_residuals_{target_item}", fig=fig_sr)
plt.show()

# ---- plot 3: point forecast for next `horizon` bars ----
horizon = garch_holdout
point_fc = GARCHForecast(garch_fit, horizon=horizon, scale=garch_scale)
print("\nHead of point forecast:")
print(point_fc.head())

last_train_ts = train_returns.index[-1]
future_idx = pd.date_range(
    start=last_train_ts + (train_returns.index[-1] - train_returns.index[-2]),
    periods=horizon, freq=(train_returns.index[-1] - train_returns.index[-2])
)

fig_pf, ax_pf = plt.subplots(figsize=(12, 4))
ax_pf.plot(cond_vol.iloc[-horizon * 2:], color='#FFD34F',
           linewidth=1.0, label='In-sample σ')
ax_pf.plot(future_idx, point_fc['volatility'].values, color='red',
           linewidth=1.2, label='Forecast σ')
ax_pf.set_title(f"GARCH Volatility Forecast (horizon={horizon})")
ax_pf.set_ylabel("σ (log-return scale)")
ax_pf.legend()
ax_pf.grid(alpha=0.3)
plt.xticks(rotation=45)
save_fig(f"garch_point_forecast_{target_item}", fig=fig_pf)
plt.show()

# ---- plot 4: Monte-Carlo density forecast fan chart over prices ----
last_price = float(garch_prices.iloc[-garch_holdout - 1])
density = GARCHForecastDensity(
    garch_fit,
    horizon       = horizon,
    n_simulations = 5000,
    scale         = garch_scale,
    last_price    = last_price,
    seed          = 42,
)

actual_future = garch_prices.iloc[-garch_holdout:]

fig_fan, ax_fan = plt.subplots(figsize=(12, 5))
# recent context
ctx = garch_prices.iloc[-garch_holdout * 2:-garch_holdout]
ax_fan.plot(ctx.index, ctx.values, color='white', linewidth=1.0, label='History')
# confidence bands from simulated price paths
q = density['quantiles']
price_q = {
    level: last_price * np.exp(np.cumsum(np.quantile(density['return_paths'], level, axis=0)))
    for level in (0.05, 0.25, 0.5, 0.75, 0.95)
}
# Use the simulated price paths directly for quantiles (more accurate than
# accumulating return quantiles, which is not the same distribution)
price_paths = density['price_paths']
p05 = np.quantile(price_paths, 0.05, axis=0)
p25 = np.quantile(price_paths, 0.25, axis=0)
p50 = np.quantile(price_paths, 0.50, axis=0)
p75 = np.quantile(price_paths, 0.75, axis=0)
p95 = np.quantile(price_paths, 0.95, axis=0)

ax_fan.fill_between(actual_future.index, p05, p95, color='#5DD4FF',
                    alpha=0.15, label='5-95% band')
ax_fan.fill_between(actual_future.index, p25, p75, color='#5DD4FF',
                    alpha=0.30, label='25-75% band')
ax_fan.plot(actual_future.index, p50, color='#5DD4FF',
            linewidth=1.2, label='Simulated median')
ax_fan.plot(actual_future.index, actual_future.values, color='#FFD34F',
            linewidth=1.2, label='Actual')
ax_fan.set_title(
    f"GARCH Monte-Carlo Density Forecast — {item_name(target_item)} "
    f"(n_sims={density['return_paths'].shape[0]}, horizon={horizon})"
)
ax_fan.set_ylabel("Price (GP)")
ax_fan.legend(loc='upper left')
ax_fan.grid(alpha=0.3)
plt.xticks(rotation=45)
save_fig(f"garch_density_fan_{target_item}", fig=fig_fan)
plt.show()

print("\nTerminal price summary at t+horizon:")
for k, v in density['terminal_summary'].items():
    print(f"  {k:<8s}: {v:,.2f}")
# %%
