#%%
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

boss_file_path = DATA_DIR / "processed_bosstables.csv"
cointegration_path = DATA_DIR / "cointegration_price_matrix.csv"

import csv
import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
import  src.utils.model_tools as tools
from src.utils.model_tools import item_name
import  src.utils.plot_tools as myplot
import  src.data_processing.data_pipeline as pipeline
import  pytz
from    data.bosstables import bosstables_list as BOSSTABLES_LIST
import  src.data_ingestion.data_fetcher as api
from    src.data_ingestion.announcements_fetcher import get_announcements
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
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

def cyclical_time(price_matrix_items:pd.DataFrame) -> tuple:
    if not isinstance(price_matrix_items.index, pd.DatetimeIndex):
        price_matrix_items.index = pd.to_datetime(price_matrix_items.index, unit='s', utc=True).tz_convert(pytz.timezone('US/Eastern'))  

    hour_time_sin = np.sin(price_matrix_items.index.minute*2*np.pi/60)
    hour_time_cos = np.cos(price_matrix_items.index.minute*2*np.pi/60)
    day_time_sin = np.sin(price_matrix_items.index.hour*2*np.pi/24)
    day_time_cos = np.cos(price_matrix_items.index.hour*2*np.pi/24)
    
    hour_trig = pd.DataFrame({'minute_sin': hour_time_sin, 'minute_cos': hour_time_cos}, index=price_matrix_items.index)
    day_trig = pd.DataFrame({'day_sin': day_time_sin, 'day_cos': day_time_cos}, index=price_matrix_items.index)

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

    merged_df = pd.merge_asof(
        left=price_matrix_items,
        right=updates_df,
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    last_update_timestamps = merged_df['last_update']

    time_difference = price_matrix_items.index - last_update_timestamps

    updates_hours_since = time_difference.dt.total_seconds() / 3600
    updates_days_since = time_difference.dt.days

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

        if cointegration_path.exists():
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

            if cointegration_path.stat().st_size == 0:
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
