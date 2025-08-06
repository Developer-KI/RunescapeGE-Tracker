#%%
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: sys.path.append(project_root)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils.model_tools as tools
import utils.plot_tools as myplot
import utils.data_pipeline as pipeline
import pytz
#%%
price_data = pipeline.data_preprocess2(read=True, interp_method='linear')
price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")
vol_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="totalvol")
volatility_index = pipeline.volatility_market(price_data, smoothing=20)[20:] #LEAKY
# converting unix to datetime
price_matrix_items.index = pd.to_datetime(price_matrix_items.index, unit='s').tz_localize(pytz.timezone('US/Eastern'))
vol_matrix_items.index = pd.to_datetime(vol_matrix_items.index, unit='s') .tz_localize(pytz.timezone('US/Eastern'))
#%%
target_item1 = np.random.choice(price_matrix_items.columns)
target_item2 = np.random.choice(price_matrix_items.columns)

price_corr = price_matrix_items.corr() 
vol_corr = vol_matrix_items.corr()
np.fill_diagonal(price_corr.values, np.nan)  
np.fill_diagonal(vol_corr.values, np.nan)

timestamps_unix = price_matrix_items.index
timestamps_datetime = pd.to_datetime(timestamps_unix, unit='s', utc=True)
timestamps_datetime_series = pd.Series(timestamps_datetime.values, index=timestamps_datetime)

base_value = 100
base_prices = price_matrix_items.iloc[0,:]
price_ratio = price_matrix_items.div(base_prices, axis=1)
mean_ratio = price_ratio.mean(axis=1) 
market_index_equal_weight = mean_ratio * base_value

vprice_matrix = price_matrix_items * vol_matrix_items
sum_vprice_matrix = vprice_matrix.sum(axis=1)
vprice_ratio = sum_vprice_matrix/sum_vprice_matrix[0]
market_index_volume_weight = base_value * vprice_ratio

equal_market_item_corr = price_matrix_items.corrwith(market_index_equal_weight)
vprice_market_item_corr = price_matrix_items.corrwith(market_index_volume_weight)
# %% Price History
myplot.plot_features(price_matrix_items[target_item1],start= '2025-05-10',end= '2025-05-11')
#%% Market Indices
myplot.plot_features(market_index_equal_weight)
#%%
myplot.plot_features(market_index_volume_weight)
# %% Volume History 
myplot.plot_features(vol_matrix_items[target_item1],start= '2025-05-10',end= '2025-05-11')
# %% Correlations
sns.heatmap(price_matrix_items.iloc[:,1:10].corr(), annot=True, cmap='coolwarm', fmt=".2f")
#%%
sns.heatmap(vol_matrix_items.iloc[:,15:30].corr(), annot=True, cmap='coolwarm', fmt=".2f")
#%% Item Beta
myplot.plot_item_market_divergence(price_matrix_items, 2, market_index_equal_weight)
#%%
myplot.plot_item_market_divergence(price_matrix_items, 2, market_index_equal_weight)
print(f'Beta: {tools.beta(price_matrix_items, target_item1, market_index_equal_weight)}')
#%% Pair Divergence
myplot.plot_feature_divergence(price_matrix_items[target_item1], price_matrix_items[target_item2], 'z', window=60,start= '2025-05-10',end= '2025-05-11')
#%%
price_corr = price_matrix_items[target_item1].rolling(20).corr(price_matrix_items[target_item2])
#1-period returns
returns_item1 = price_matrix_items[target_item1].pct_change()
returns_item2 = price_matrix_items[target_item2].pct_change()
#hourly-returns
returns_item1_hourly = price_matrix_items[target_item1].resample('h').last().pct_change()
returns_item2_hourly = price_matrix_items[target_item2].resample('h').last().pct_change()
#daily-returns
returns_item1_daily = price_matrix_items[target_item1].resample('d').last().pct_change()
returns_item2_daily = price_matrix_items[target_item2].resample('d').last().pct_change()

# %%
