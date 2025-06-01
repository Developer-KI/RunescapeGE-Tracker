# %% Script Init
import pandas as pd
import numpy as np
import DataPipeline as pipeline
import ModelTools as tools
import Models.RFTS as myRFTS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import Models.HMM as myHMM
# %% Aggregate features for model
price_data = pipeline.data_preprocess(read=False, interp_method='nearest')
price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")
vol_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="totalvol")

target_item = 12934
price_items_reg = price_matrix_items[[target_item]].iloc[20:]
vol_items_reg = vol_matrix_items[[target_item]].iloc[20:]
price_items_reg.columns = [f'{target_item}']
vol_items_reg.columns = [f'{target_item}_vol']
volatility_index = pipeline.volatility_market(price_data, smoothing=20)[20:] #LEAKY
reg_data = pd.concat([price_items_reg, vol_items_reg, volatility_index], axis=1)

df_time = tools.target_time_features(reg_data, f'{target_item}', 10) 
df_roll = tools.target_rolling_features(reg_data, f'{target_item}', 20) #LEAKY
df = pd.merge(df_time, df_roll, on='timestamp', how='inner').dropna()
df_mod=df[['12934','lag1','lag2','lag3','lag4']] #Make sure target is always the first column
#%% Run the optimized RFTS
newoptim, best_test_idx = myRFTS.RFTSOptim(df_mod,target_col=f'{target_item}',n_trials=5)
tools.plot_pred_vs_price(df_mod, model=newoptim['Best Model'],index=best_test_idx,lookback=100)
#%% Run the HMM model
myHMM.ItemThresholdHMM(price_matrix_items,207)
