# %% Script Init
import pandas as pd
import numpy as np
import DataPipeline as pipeline
import ModelTools as tools
import Models.RFTS as myRFTS
import Models.XGB as myXGB
import Models.HMM as myHMM
import Models.EXPSmooth as myEXPS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor
#
price_data = pipeline.data_preprocess2(read=True, interp_method='linear')

price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")
vol_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="totalvol")
#%%
target_item = np.random.choice(price_matrix_items.columns)
target_item = 1215
price_items_reg = price_matrix_items[[target_item]].iloc[20:]
vol_items_reg = vol_matrix_items[[target_item]].iloc[20:]
price_items_reg.columns = [f'{target_item}']
vol_items_reg.columns = [f'{target_item}_vol']
volatility_index = pipeline.volatility_market(price_data, smoothing=20)[20:] #LEAKY
reg_data = pd.concat([price_items_reg, vol_items_reg, volatility_index], axis=1)

df_time = tools.target_time_features(reg_data, f'{target_item}', 100) 
df_roll = tools.target_rolling_features(reg_data, f'{target_item}', 20) #LEAKY
df = pd.merge(df_time, df_roll, on='timestamp', how='inner').dropna()
#
df_mod=df[[f'{target_item}','lag1','lag2','lag3']] #Make sure target is always the first column

#Fix outlier detection
#models still train on outliers !!!!
#%% Run a RFTS model
RFTSmodel, holdout_pred_rfts, outlier_rfts = myRFTS.RFTS(df_mod, target_col=f'{target_item}', holdout=500, outlier_threshold=7, max_depth=5, n_estimators=400, min_samples_leaf=20)
tools.plot_pred_vs_price(df_mod, model=RFTSmodel,lookback=9800,holdout_pred=holdout_pred_rfts,fill_outliers=outlier_rfts)
#%% Run a XBG model
XGBModel, holdout_pred_xgb, outlier_xgb = myXGB.XGB(df_mod, target_col=f'{target_item}', holdout=500,max_depth=5, n_estimators=400)
tools.plot_pred_vs_price(df_mod, model=XGBModel,lookback=9800,holdout_pred=holdout_pred_xgb, fill_outliers=outlier_xgb)
#%% Run an EXPS model **possible axis issue? (disappeared)
EXPSmodel, pred_exps, extreme_exps, outlier_exps = myEXPS.EXPSmooth(df_mod,holdout=200)
tools.plot_pred_vs_price(df_mod,model=EXPSmodel,lookback=1000,holdout_pred=pred_exps)
#%% Run the optimized RFTS
optim, optimparam,best_test_idx = myRFTS.RFTSOptim(df_mod,target_col=f'{target_item}',n_trials=15)
#%%
tools.plot_pred_vs_price(df_mod, model=optim,best_index=best_test_idx,lookback=0)
#%%
tools.test_train_error(df_mod, param='colsample_bytree', 
                       exclude_param={'learning_rate':0.03, 'max_depth':5,'subsample':0.4,'min_child_weight':5}, 
                       model_class=myXGB.XGBRegressor,
                       param_range=(0.1,1,0.1))
#%%