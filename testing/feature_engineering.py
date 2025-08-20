#%%
import  os
import  sys

# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to reach the project root
project_root = os.path.join(current_dir, '..')
# Add the project root to the system path
if project_root not in sys.path:
    sys.path.append(project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
boss_file_path = os.path.join(project_root, 'data', 'processed_bosstables.csv')

import  numpy as np
import  pandas as pd
#import  seaborn as sns
import  matplotlib.pyplot as plt
import  utils.model_tools as tools
import  utils.plot_tools as myplot
import  utils.data_pipeline as pipeline
import  pytz
from    data.bosstables import bosstables_list as BOSSTABLES_LIST
import  utils.api_fetcher as api
from    utils.announcements_fetcher import get_announcements
#%% General Prices
price_data = pipeline.data_preprocess2(read=True, write=False, interp_method='linear', filter_volume=True, filter_threshold=0.98)
price_data = price_data.set_index('timestamp')
def item_data(price_data, datetime: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    #unix time conversion & localization
    if datetime:
        #possible silent failure inconsistent feature localization, check pipeline, site/api
        price_data.index = pd.to_datetime(price_data.index, unit='s', utc=True).tz_convert(pytz.timezone('US/Eastern')) 
    price_matrix_items = price_data.pivot(columns="item_id", values="wprice") #hmmmm
    vol_matrix_items = price_data.pivot(columns="item_id", values="totalvol")
    return price_matrix_items, vol_matrix_items

#Boss Items
def boss_data(datetime: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    boss_ids= set([item for sublist in BOSSTABLES_LIST for item in sublist]) #set conversion drops dupelicates
    boss_data = pipeline.data_explicit_preprocess( #refine argument handling
        boss_ids, 
        read_path= boss_file_path
    )
    boss_matrix_items = boss_data.pivot(index="timestamp", columns="item_id", values="wprice")
    if datetime:
        boss_matrix_items.index = pd.to_datetime(boss_matrix_items.index, unit='s', utc=True).tz_convert('US/Eastern')
    return boss_data, boss_matrix_items
# bandos_index = tools.create_item_index(boss_matrix_items,[
# tools.item_name("Bandos chestplate"),
# tools.item_name("Bandos tassets"),
# tools.item_name("Bandos boots"),
# tools.item_name("Bandos hilt"),
# tools.item_name("Godsword shard 1"),
# tools.item_name("Godsword shard 2"),
# tools.item_name("Godsword shard 3"),
# ], 'equal',100)

#Correlations
def item_corr(price_matrix_items: pd.DataFrame, vol_matrix_items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_corr = price_matrix_items.corr() 
    vol_corr = vol_matrix_items.corr()
    np.fill_diagonal(price_corr.values, np.nan)  
    np.fill_diagonal(vol_corr.values, np.nan)
    return price_corr, vol_corr

#Indices
def market_indices(price_matrix_items: pd.DataFrame, vol_matrix_items: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    market_index_volume_weight = tools.create_item_index(
        [price_matrix_items,vol_matrix_items],
        list(price_matrix_items.columns),
        type='vprice',base_value=100)

    market_index_equal_weight = tools.create_item_index(
        price_matrix_items,
        list(price_matrix_items.columns),
        type='equal',base_value=100)
    return market_index_equal_weight, market_index_volume_weight

def market_item_corr(market_index: pd.Series, price_matrix_items) -> pd.Series:
    index_item_corr = price_matrix_items.corrwith(market_index)
    return index_item_corr

#indexes for rune,log,herb,food,metal

# %% Price History
#price_corr_items = price_matrix_items[target_item1].rolling(20).corr(price_matrix_items[target_item2])
#1-period returns
#returns_item1 = price_matrix_items[target_item1].pct_change()
#returns_item2 = price_matrix_items[target_item2].pct_change()
#hourly-returns
#returns_item1_hourly = tools.calculate_returns(price_matrix_items[target_item1], '1h')
#returns_item2_hourly = tools.calculate_returns(price_matrix_items[target_item2], '1h')
#daily-returns
#returns_item1_hourly = tools.calculate_returns(price_matrix_items[target_item1], '1D')
#returns_item2_hourly = tools.calculate_returns(price_matrix_items[target_item2], '1D')
# cyclical time features
def cyclical_time(price_matrix_items:pd.DataFrame) -> tuple:
    if not isinstance(price_matrix_items.index, pd.DatetimeIndex):
        price_matrix_items.index = pd.to_datetime(price_matrix_items.index, unit='s', utc=True).tz_convert(pytz.timezone('US/Eastern'))  
    hour_time_sin = np.sin(price_matrix_items.index.minute*2*np.pi/60)
    hour_time_cos = np.cos(price_matrix_items.index.minute*2*np.pi/60)
    day_time_sin = np.sin(price_matrix_items.index.hour*2*np.pi/24)
    day_time_cos = np.cos(price_matrix_items.index.hour*2*np.pi/24)
    hour_trig = np.vstack((hour_time_sin, hour_time_cos)).T
    day_trig = np.vstack((day_time_sin, day_time_cos)).T
    return hour_time_sin, hour_time_cos, day_time_sin, day_time_cos, hour_trig, day_trig

def updates_announcements() -> tuple[pd.DatetimeIndex, np.ndarray, pd.Series]:
    #update info: 6:30 EST every Wednesday
    update_dates = pd.date_range(
        start='2015-03-28 11:30',
        end=pd.Timestamp.now(),
        freq='W-WED', # weekly wednesday
        tz='Europe/London' # Specify the British timezone
        ).tz_convert('US/Eastern') #back to EST
    
                
    announcements = get_announcements()['timestamp'].dt.tz_localize('US/Eastern').dt.date #UTC assumption !!!
    return update_dates, updates_hours_since, updates_days_since, announcements