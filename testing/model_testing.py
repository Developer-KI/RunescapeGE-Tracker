#%% 
import  os, sys
#For relative pathing
# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to reach the project root
project_root = os.path.join(current_dir, '..')
# Add the project root to the system path
if project_root not in sys.path:
    sys.path.append(project_root)
#%%
import  pandas as pd, numpy as np
from    utils.model_tools import (
    target_rolling_features, 
    create_feature_lags, 
    volatility_market, 
    item_name, 
    create_item_index
)
import  utils.plot_tools as myplot
from    models import (
    rfts as myRFTS, 
    xgb as myXGB, 
    hmm as myHMM, 
    exp_smoothing as myEXPS
    )
from    matplotlib import pyplot as plt
from    matplotlib import ticker as mticker
import  feature_engineering as get
import utils.outlier_detection as outlier
from utils.model_tools import item_name
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
RFTS_residuals  = myplot.plot_residuals(master, rfts_model)
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
predictions_xgb = rfts_model.predict(master.drop(f'{target_item}', axis=1))
myplot.plot_pred_vs_price(master, predictions=predictions_xgb, holdout_pred_n=holdout, lookback=5000,holdout_pred=holdout_pred_xgb, fill_outliers=train_outliers_xgb+holdout_outliers_xgb)
feature_importances = pd.Series(xgb_model.feature_importances_, index=master.columns[1:]).sort_values(ascending=False)
print(feature_importances)
#%% Run an EXPS model **possible axis issue? (disappeared)
EXPSmodel, pred_exps = myEXPS.EXPSmooth(master,holdout=200) #omitting extreme_exps, outlier_exps
myplot.plot_pred_vs_price(master,model=EXPSmodel,lookback=1000,holdout_pred=pred_exps)
#%% Run the optimized RFTS
optim, optimparam,best_test_idx = myRFTS.RFTSOptim(master,target_col=f'{target_item}',n_trials=50, holdout=500)
#%%
myplot.plot_pred_vs_price(master, model=optim,best_index=best_test_idx,lookback=0)
#%%
myplot.test_train_error(master, param='max_depth', 
                       exclude_param={}, 
                       model_class=myRFTS.RandomForestRegressor,
                       param_range=())
# %%
plt.figure(figsize=(10,5))
plt.plot(range(len(cv_mae_rfts)),cv_mae_rfts, label='Test')
plt.plot(range(len(train_cv_mae_rfts)),train_cv_mae_rfts, label='Train')
plt.legend()
plt.show()
# %%
plt.figure(figsize=(10,5))
plt.plot(range(len(cv_mae_xgb)),cv_mae_xgb, label='Test')
plt.plot(range(len(train_cv_mae_xgb)),train_cv_mae_xgb, label='Train')
plt.legend()
plt.show()
# %% ------------------------------------------------
myplot.plot_item_market_divergence(price_matrix_items, 1603, equal_index, '10min')
# %% -------------------------------------------------
full_y = price_matrix_items[1603]
volume = vol_matrix_items[1603]
window=100
threshold = 1

fig, ax = myplot.plot_features(full_y, start='2025-08-4', end='2025-08-11')
log_return = np.log(full_y / full_y.shift(1))

ratio = np.abs(log_return) / volume

rolling_mean_ratio = ratio.rolling(window=window).mean()
rolling_std_ratio = ratio.rolling(window=window).std()

outlier_threshold = rolling_mean_ratio + rolling_std_ratio * threshold

# %%
fig, ax = myplot.plot_features(ratio, start='2025-08-4',end='2025-08-11')
ax.plot(outlier_threshold.loc['2025-08-4':'2025-08-11'], label='threshold')
fig.legend()
# %%
