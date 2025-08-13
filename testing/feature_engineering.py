#%%
import  os
import  sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
if project_root not in sys.path: sys.path.append(project_root)
import  numpy as np
import  pandas as pd
# import  seaborn as sns
import  matplotlib.pyplot as plt
import  utils.model_tools as tools
import  utils.plot_tools as myplot
import  utils.data_pipeline as pipeline
import  pytz
from    data.bosstables import bosstables_list as BOSSTABLES_LIST
import  utils.api_fetcher as api
from    utils.announcements_fetcher import get_announcements
#%% General Prices
price_data = pipeline.data_preprocess2(read=True, write=False, interp_method='linear', filter_volume=True)
#unix time conversion & localization
price_data = price_data.set_index('timestamp')
#possible silent failure incocnsistent feature localization, check pipeline, site/api
price_data.index = pd.to_datetime(price_data.index, unit='s', utc=True).tz_convert(pytz.timezone('US/Eastern')) 
price_matrix_items = price_data.pivot(columns="item_id", values="wprice") #hmmmm
price_matrix_items2 = price_data.pivot(columns="item_id", values=["wprice",'avgHighPrice'])
vol_matrix_items = price_data.pivot(columns="item_id", values="totalvol")
#Boss Items
boss_ids= set([item for sublist in BOSSTABLES_LIST for item in sublist]) #set conversion drops dupelicates
boss_data = pipeline.data_explicit_preprocess( #refine argument handling
    boss_ids, 
    read_path='../data/processed_bosstables.csv'
    )
boss_matrix_items = boss_data.pivot(index="timestamp", columns="item_id", values="wprice")
boss_matrix_items.index = pd.to_datetime(boss_matrix_items.index, unit='s', utc=True).tz_convert('US/Eastern')
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
#Indices
market_index_volume_weight = tools.create_item_index(
    [price_matrix_items,vol_matrix_items],
    list(price_matrix_items.columns),
    type='vprice',base_value=100)

market_index_equal_weight = tools.create_item_index(
    price_matrix_items,
    list(price_matrix_items.columns),
    type='equal',base_value=100)

equal_market_item_corr = price_matrix_items.corrwith(market_index_equal_weight)
vprice_market_item_corr = price_matrix_items.corrwith(market_index_volume_weight)

#indexes for rune,log,herb,food,metal

# %% Price History
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
#cyclical time features
hour_time_sin = np.sin(price_matrix_items.index.minute*2*np.pi/60)
hour_time_cos = np.cos(price_matrix_items.index.minute*2*np.pi/60)
day_time_sin = np.sin(price_matrix_items.index.hour*2*np.pi/24)
day_time_cos = np.cos(price_matrix_items.index.hour*2*np.pi/24)
#update info: 6:30 EST every Wednesday
update_dates = pd.date_range(
    start='2015-03-28 11:30',
    end=pd.Timestamp.now(),
    freq='W-WED', # weekly wednesday
    tz='Europe/London' # Specify the British timezone
).tz_convert('US/Eastern') #back to EST

announcements = set(get_announcements()['timestamp'].dt.tz_localize('US/Eastern').dt.date)
# %%

