from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

import pandas as pd
import json
import numpy as np
from typing import List

def data_explicit_preprocess(
    items:          list,
    read_path:      Path|str|None = None,
    file_path = DATA_DIR / "data.csv",
    write_name:     str|None = None,
    interp_method:  str = 'linear' 
) -> pd.DataFrame:
    if type(read_path) is str or Path:
        preprocessed_pricedata = pd.read_csv(
                read_path, 
                names = [
                'timestamp', 'item_id', 'avgHighPrice', 'highPriceVolume',
                'avgLowPrice', 'lowPriceVolyme', 'totalvol', 'wprice'
                ]
            )
        print(f"Successfully loaded processed data from {read_path}")
        return preprocessed_pricedata

    elif not isinstance(read_path, (str, type(None))): raise ValueError("Argument read_path should be of None or string") 

    raw_pricedata =  pd.read_csv(
            file_path, 
            names = [
                'item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice',
                'lowPriceVolume', 'timestamp', 'totalvol', 'wprice'      
                ]
            )
    raw_pricedata = raw_pricedata.sort_values(by=['item_id', 'timestamp']).reset_index(drop=True)
    
    # Remove duplicate (item_id, timestamp) entries, keeping the last observed value
    raw_pricedata.drop_duplicates(subset=['item_id', 'timestamp'], keep='last', inplace=True)
    
    initial_rows = len(raw_pricedata)
    if len(raw_pricedata) < initial_rows:
        print(f"Removed {initial_rows - len(raw_pricedata)} duplicate (item_id, timestamp) entries from raw data.")
    
    raw_pricedata = raw_pricedata[raw_pricedata['item_id'].isin(items)]

    # --- Step 1: Create a COMPLETE GRID of (timestamp, item_id) combinations ---
    all_timestamps = np.sort(raw_pricedata['timestamp'].unique()) # Use np.sort for efficiency
    all_item_ids = raw_pricedata['item_id'].unique()

    full_index = pd.MultiIndex.from_product([all_timestamps, all_item_ids], names=['timestamp', 'item_id'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Merge the raw data onto the full grid, introducing NaNs for missing combinations
    processed_pricedata = pd.merge(full_df, raw_pricedata, on=['timestamp', 'item_id'], how='left')

    print(f"\nNaNs after creating full grid and left merging (expected for initially missing combos):\n{processed_pricedata.isnull().sum()}")

    # Fill NA volumes with 0 prior to calculation to avoid NaN propagation
    processed_pricedata['highPriceVolume'] = processed_pricedata['highPriceVolume'].fillna(0)
    processed_pricedata['lowPriceVolume'] = processed_pricedata['lowPriceVolume'].fillna(0)
    
    processed_pricedata['totalvol'] = processed_pricedata['highPriceVolume'] + processed_pricedata['lowPriceVolume']

    processed_pricedata['avgHighPrice_calc'] = processed_pricedata.groupby('item_id')['avgHighPrice'].transform(lambda x: x.ffill())
    processed_pricedata['avgLowPrice_calc'] = processed_pricedata.groupby('item_id')['avgLowPrice'].transform(lambda x: x.ffill())

    processed_pricedata['wprice'] = np.where( # Use np.where
        processed_pricedata['totalvol'] == 0,
        np.nan, # Use np.nan for explicit missing values
        (processed_pricedata['highPriceVolume'] / processed_pricedata['totalvol']) * \
        (processed_pricedata['avgHighPrice_calc'] - processed_pricedata['avgLowPrice_calc']) + \
        processed_pricedata['avgLowPrice_calc']
    )
    
    # Drop the temporary calculation columns
    processed_pricedata.drop(columns=['avgHighPrice_calc', 'avgLowPrice_calc'], inplace=True)

    # --- Step 3: Grouped Interpolation (forward-only) and Forward Fill for all relevant columns ---
    print("\nStarting grouped forward-only interpolation and forward fills...")
    
    cols_to_fill = ['avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'totalvol', 'wprice']
    
    for col in cols_to_fill:
        # Interpolate missing values within each item_id group, looking only forward.
        # Then, ffill any remaining NaNs (especially leading ones or those after interpolation)
        processed_pricedata[col] = processed_pricedata.groupby('item_id')[col].transform(
            lambda x: x.interpolate(method=interp_method, limit_direction='forward').ffill()
        )

    processed_pricedata.ffill(inplace=True)

    print(f"\nNaNs after grouped forward-only interpolation and forward fills: \n{processed_pricedata.isnull().sum()}")
    print(f"Total NaNs in processed_pricedata (may be non-zero for leading NaNs): {processed_pricedata.isnull().sum().sum()}")
    
    # Reorder columns to ensure consistent output structure, especially for saving and future reading
    final_columns_order = ['timestamp', 'item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'totalvol', 'wprice']
    processed_pricedata = processed_pricedata[final_columns_order]
    
    if type(write_name) is str:
        processed_pricedata.to_csv(DATA_DIR / write_name, mode='w', header=False, index=False)
    else: raise ValueError("Argument write_name should be type string or None")
    return processed_pricedata

def data_preprocess2(
    read:               bool = True,
    filepath = DATA_DIR,
    read_path = DATA_DIR / "processed_data.csv",
    write:              bool = False, 
    interp_method:      str = 'linear', 
    filter_volume:      bool = True,
    filter_threshold:   float = 0.99
) -> pd.DataFrame:
   
    if read:
        try:
            df = pd.read_csv(
                read_path,
                names= ['timestamp', 'item_id', 'avgHighPrice',
                        'highPriceVolume', 'avgLowPrice', 'lowPriceVolume',
                        'totalvol', 'wprice'
                    ]
            )
            print(f"Successfully loaded processed data from {read_path}")
            return df
        except FileNotFoundError:
            print(f"Processed data not found at {read_path}. Proceeding with raw data processing.")
            pass # Fall through to process raw data if file not found
        except Exception as e:
            print(f"Error reading processed data from {read_path}: {e}. Proceeding with raw data processing.")
            pass

    raw_pricedata = pd.read_csv(filepath / "data.csv", names=['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp'])

    raw_pricedata = raw_pricedata.sort_values(by=['item_id', 'timestamp']).reset_index(drop=True)

    initial_rows = len(raw_pricedata)
    raw_pricedata.drop_duplicates(subset=['item_id', 'timestamp'], keep='last', inplace=True)
    if len(raw_pricedata) < initial_rows:
        print(f"Removed {initial_rows - len(raw_pricedata)} duplicate (item_id, timestamp) entries from raw data.")

    try:
        if filter_volume:
            with open(filepath / "data_properties.txt", "r") as file:
                lines = file.readlines()
        else:
            with open(filepath / "data_properties_full.txt", "r") as file:
                lines = file.readlines()
        series_length = int(lines[2].replace("\n", ""))
    except FileNotFoundError:
        print(f"Error: data_properties.txt not found at {filepath}. Cannot determine series_length.")
        series_length = raw_pricedata['timestamp'].nunique() # Fallback to number of unique timestamps
        print(f"Using unique timestamp count as series_length: {series_length}")
    except IndexError:
        print(f"Error: data_properties.txt at {filepath} does not contain expected series_length on the second line.")
        series_length = raw_pricedata['timestamp'].nunique()
        print(f"Using unique timestamp count as series_length: {series_length}")

    print(f"Raw data loaded. Initial NaNs (should be 0 for raw inputs, unless explicit in CSV):\n{raw_pricedata.isnull().sum()}")

    group_raw_pricedata = raw_pricedata.groupby('item_id').nunique()

    if filter_volume:
        filtered_indexes = group_raw_pricedata[group_raw_pricedata['timestamp'] < filter_threshold * series_length].index
        raw_pricedata = raw_pricedata[~raw_pricedata['item_id'].isin(filtered_indexes)]
    print(f"\nAfter filtering: {len(raw_pricedata['item_id'].unique())} item(s) remaining.")
    
    all_timestamps = np.sort(raw_pricedata['timestamp'].unique()) # Use np.sort for efficiency
    all_item_ids = raw_pricedata['item_id'].unique()

    full_index = pd.MultiIndex.from_product([all_timestamps, all_item_ids], names=['timestamp', 'item_id'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    processed_priced_data = pd.merge(full_df, raw_pricedata, on=['timestamp', 'item_id'], how='left')

    print(f"\nNaNs after creating full grid and left merging (expected for initially missing combos):\n{processed_priced_data.isnull().sum()}")

    processed_priced_data['highPriceVolume'] = processed_priced_data['highPriceVolume'].fillna(0)
    processed_priced_data['lowPriceVolume'] = processed_priced_data['lowPriceVolume'].fillna(0)
    
    processed_priced_data['totalvol'] = processed_priced_data['highPriceVolume'] + processed_priced_data['lowPriceVolume']

    processed_priced_data['avgHighPrice_calc'] = processed_priced_data.groupby('item_id')['avgHighPrice'].transform(lambda x: x.ffill())
    processed_priced_data['avgLowPrice_calc'] = processed_priced_data.groupby('item_id')['avgLowPrice'].transform(lambda x: x.ffill())

    processed_priced_data['wprice'] = np.where( # Use np.where
        processed_priced_data['totalvol'] == 0,
        np.nan, # Use np.nan for explicit missing values
        (processed_priced_data['highPriceVolume'] / processed_priced_data['totalvol']) * \
        (processed_priced_data['avgHighPrice_calc'] - processed_priced_data['avgLowPrice_calc']) + \
        processed_priced_data['avgLowPrice_calc']
    )
    
    processed_priced_data.drop(columns=['avgHighPrice_calc', 'avgLowPrice_calc'], inplace=True)

    print("\nStarting grouped forward-only interpolation and forward fills...")
    
    cols_to_fill = ['avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'totalvol', 'wprice']
    
    for col in cols_to_fill:
        # Interpolate missing values within each item_id group, looking only forward.
        # Then, ffill any remaining NaNs (especially leading ones or those after interpolation)
        processed_priced_data[col] = processed_priced_data.groupby('item_id')[col].transform(
            lambda x: x.interpolate(method=interp_method, limit_direction='forward').ffill()
        )

    processed_priced_data.ffill(inplace=True)

    print(f"\nNaNs after grouped forward-only interpolation and forward fills: \n{processed_priced_data.isnull().sum()}")
    print(f"Total NaNs in processed_priced_data (may be non-zero for leading NaNs): {processed_priced_data.isnull().sum().sum()}")
    
    final_columns_order = ['timestamp', 'item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'totalvol', 'wprice']
    processed_priced_data = processed_priced_data[final_columns_order]

    if write:
        processed_priced_data.to_csv(filepath / "processed_data.csv", mode='w', header=False, index=False)
        print(f"Processed data saved to {filepath / 'processed_data.csv'}")
    
    return processed_priced_data

def data_preprocess_deprecated(read: bool, filepath = DATA_DIR, read_path = DATA_DIR / "processed_data.csv", write: bool = False, interp_method: str = 'linear') -> pd.DataFrame:
    if read:
        df = pd.read_csv(read_path, names=['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp', 'totalvol', 'wprice'])
        return df

    raw_pricedata = pd.read_csv(filepath / "data.csv", names=['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp'])
    with open(filepath / "data_properties.txt", "r") as file:
        lines = file.readlines()
        file.close()
    series_length = int(lines[1].replace("\n", ""))

    group_raw_pricedata = raw_pricedata.groupby('item_id').nunique()
    filtered_indexes = group_raw_pricedata[group_raw_pricedata['timestamp'] != series_length].index
    raw_pricedata = raw_pricedata[~raw_pricedata['item_id'].isin(filtered_indexes)]
    
    # interpolate missing values
    processed_priced_data = raw_pricedata.interpolate(method=interp_method)
    
    processed_priced_data['totalvol'] = processed_priced_data['highPriceVolume'] + processed_priced_data['lowPriceVolume']
    processed_priced_data['wprice'] = (processed_priced_data['highPriceVolume']/processed_priced_data['totalvol']) * (processed_priced_data['avgHighPrice'] - processed_priced_data['avgLowPrice']) + processed_priced_data['avgLowPrice']

    #Saving output
    if write:
        processed_priced_data.to_csv(filepath / "processed_data.csv", mode='w', header=False, index=False)

    return processed_priced_data

def alchemy_preprocess(read: bool = True, filepath = DATA_DIR, read_path = DATA_DIR / "alchemy_data.csv", write: bool = False) -> pd.DataFrame:
    ### Read has higher priority than write
    if read:
        df = pd.read_csv(read_path, names=['item', 'price'], index_col=0)
        return df

    with open(filepath / "namealchemy.json", "r") as file:
        alc_data = json.load(file)
        file.close()
    with open(filepath / "nameID.json", "r") as file:
        name_data = json.load(file)
        file.close()

    high_alchemy = pd.DataFrame(list(alc_data.items()), columns=["item", "price"])
    nameID = pd.DataFrame(list(name_data.items()), columns=["item", "item_id"])

    #Processing tables
    reference = pd.merge(nameID, high_alchemy, on="item", how="inner")
    reference = reference.drop([0,1])
    reference.set_index('item_id', inplace=True)

    #Saving output
    if write:
        reference.to_csv(filepath / "alchemy_data.csv", mode='w', header=False, index=True)

    return reference

if __name__ == "__main__":
    test = data_preprocess2(read=False, write=True) #run in interactive

