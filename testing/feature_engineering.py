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
cointegration_path = os.path.join(project_root, 'data', 'cointegration_price_matrix.csv')

import csv
import  numpy as np
import  pandas as pd
#import  seaborn as sns
import  matplotlib.pyplot as plt
import  utils.model_tools as tools
from utils.model_tools import item_name
import  utils.plot_tools as myplot
import  utils.data_pipeline as pipeline
import  pytz
from    data.bosstables import bosstables_list as BOSSTABLES_LIST
import  utils.api_fetcher as api
from    utils.announcements_fetcher import get_announcements
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
#from arch.unitroot.cointegration import phillips_ouliaris
from itertools import combinations
#%% General Prices
price_data = pipeline.data_preprocess2(read=True, write=False, interp_method='linear', filter_volume=True, filter_threshold=0.98)
price_data = price_data.set_index('timestamp')
#%%
def item_data(price_data, wprice: bool = True, datetime: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    #unix time conversion & localization
    if datetime:
        #possible silent failure inconsistent feature localization, check pipeline, site/api
        price_data.index = pd.to_datetime(price_data.index, unit='s', utc=True).tz_convert(pytz.timezone('US/Eastern')) 
    if wprice:
        price_matrix_items = price_data.pivot(columns="item_id", values="wprice") #hmmmm
    else:
        price_matrix_items = price_data.pivot(columns="item_id", values=["avgLowPrice","avgHighPrice"])
    vol_matrix_items = price_data.pivot(columns="item_id", values="totalvol")
    return price_matrix_items, vol_matrix_items

#Boss Items
def boss_data(datetime: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    boss_ids= set([item for sublist in BOSSTABLES_LIST for item in sublist]) #set conversion drops dupelicates
    boss_data = pipeline.data_explicit_preprocess( #refine argument handling
        boss_ids, 
        read_path= boss_file_path
    )
    boss_matrix_items = boss_data.pivot(index="timestamp", columns="item_id", values="wprice")
    boss_vol_matrix_items = boss_data.pivot(index="timestamp", columns="item_id", values="totalvol")
    if datetime:
        boss_matrix_items.index = pd.to_datetime(boss_matrix_items.index, unit='s', utc=True).tz_convert('US/Eastern')
    return boss_data, boss_matrix_items, boss_vol_matrix_items
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
def item_corr(price_matrix_items: pd.DataFrame|None = None, vol_matrix_items: pd.DataFrame|None = None) -> tuple[pd.DataFrame|None, pd.DataFrame|None]|pd.DataFrame:
    if price_matrix_items is None and vol_matrix_items is None:
        raise ValueError("At least one DataFrame must be provided.")
    price_corr = None
    vol_corr = None
    
    if price_matrix_items is not None:
        price_corr = price_matrix_items.corr() 
        np.fill_diagonal(price_corr.values, np.nan) 
        
    if vol_matrix_items is not None:
        vol_corr = vol_matrix_items.corr()
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

# Price History
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

def updates_announcements(price_matrix_items: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series, pd.Series]:
    #update info: 6:30 EST every Wednesday
    update_dates = pd.date_range(
        start='2015-03-28 11:30',
        end=pd.Timestamp.now(),
        freq='W-WED', # weekly wednesday
        tz='Europe/London' # Specify the British timezone
        ).tz_convert('US/Eastern') #back to EST
    updates_df = pd.DataFrame(
        {'last_update': update_dates},
        index=update_dates
    )

    # Use pd.merge_asof to find the most recent update for each price point
    merged_df = pd.merge_asof(
        left=price_matrix_items,
        right=updates_df,
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
     # The merged DataFrame now has a 'last_update' column with the correct timestamps
    last_update_timestamps = merged_df['last_update']

    # Calculate the time difference
    time_difference = price_matrix_items.index - last_update_timestamps

    # Now you can correctly calculate hours and days
    updates_hours_since = time_difference.dt.total_seconds() / 3600
    updates_days_since = time_difference.dt.days

    # You will need to set the index of the final Series to align with your original DataFrame
    updates_hours_since.index = price_matrix_items.index
    updates_days_since.index = price_matrix_items.index

    announcements = get_announcements()['timestamp'].dt.tz_localize('US/Eastern').dt.date #UTC assumption !!!
    return update_dates, updates_hours_since, updates_days_since, announcements
#%%
def cointegration_pairs(price_matrix_items: pd.DataFrame, scrape: bool = False) -> pd.DataFrame|None:
    df = pd.read_csv(cointegration_path)
    if not scrape:
        return df
    else:
        processed_pairs = set()

        if os.path.exists(cointegration_path):
            print(f"Existing file found. Resuming from last run.")
            with open(cointegration_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader) # Skip the header row
                for row in reader:
                    if len(row) >= 2:
                        # Add the pair to our set for quick lookups
                        processed_pairs.add((row[0], row[1]))
        else:
            print("No existing file found. Starting a new run.")

        with open(cointegration_path, 'a', newline='') as f:
            writer = csv.writer(f)

            if os.path.getsize(cointegration_path) == 0:
                writer.writerow(['item1','item2','p_value'])

            for item1, item2 in combinations(price_matrix_items.columns, 2):
                if (str(item1), str(item2)) in processed_pairs or (str(item2), str(item1)) in processed_pairs:
                    continue  # Skip to the next pair

                try:
                    test_result = coint(price_matrix_items[item1], price_matrix_items[item2])
                    p_value = test_result[1]
                    print(f'{item_name(item1)} and {item_name(item2)} p-value: {p_value}')
                    writer.writerow([item1, item2, p_value])
                except Exception as e:
                    print(f"An error occurred while testing pair ({item1}, {item2}): {e}")
                    writer.writerow([item1, item2, "ERROR"])

        print("\nScript finished. All cointegration tests are now complete.")
    # %%
