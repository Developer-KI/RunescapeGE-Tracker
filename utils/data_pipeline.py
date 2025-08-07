import pandas as pd
import json
import numpy as np
from typing import List

def data_explicit_preprocess(
    items:          List[int],
    file_path:      str = '../data/data.csv', 
    interp_method:  str = 'linear', 
) -> pd.DataFrame:
    raw_pricedata = pd.read_csv(
            f'{file_path}', 
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
    processed_priced_data = pd.merge(full_df, raw_pricedata, on=['timestamp', 'item_id'], how='left')

    print(f"\nNaNs after creating full grid and left merging (expected for initially missing combos):\n{processed_priced_data.isnull().sum()}")

    # Fill NA volumes with 0 prior to calculation to avoid NaN propagation
    processed_priced_data['highPriceVolume'] = processed_priced_data['highPriceVolume'].fillna(0)
    processed_priced_data['lowPriceVolume'] = processed_priced_data['lowPriceVolume'].fillna(0)
    
    processed_priced_data['totalvol'] = processed_priced_data['highPriceVolume'] + processed_priced_data['lowPriceVolume']

    # Temporarily fill avgHighPrice/avgLowPrice for 'wprice' calculation if they are NaN at this stage.
    # We use grouped forward fill (ffill) to prevent data leakage and ensure that
    # price calculations only use information from the past.
    processed_priced_data['avgHighPrice_calc'] = processed_priced_data.groupby('item_id')['avgHighPrice'].transform(lambda x: x.ffill())
    processed_priced_data['avgLowPrice_calc'] = processed_priced_data.groupby('item_id')['avgLowPrice'].transform(lambda x: x.ffill())

    # Calculate 'wprice' (weighted price).
    # Use np.where to handle division by zero safely: if 'totalvol' is 0, 'wprice' is NaN.
    # Otherwise, calculate 'wprice' based on volumes and average prices.
    processed_priced_data['wprice'] = np.where( # Use np.where
        processed_priced_data['totalvol'] == 0,
        np.nan, # Use np.nan for explicit missing values
        (processed_priced_data['highPriceVolume'] / processed_priced_data['totalvol']) * \
        (processed_priced_data['avgHighPrice_calc'] - processed_priced_data['avgLowPrice_calc']) + \
        processed_priced_data['avgLowPrice_calc']
    )
    
    # Drop the temporary calculation columns
    processed_priced_data.drop(columns=['avgHighPrice_calc', 'avgLowPrice_calc'], inplace=True)

    # --- Step 3: Grouped Interpolation (forward-only) and Forward Fill for all relevant columns ---
    print("\nStarting grouped forward-only interpolation and forward fills...")
    
    cols_to_fill = ['avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'totalvol', 'wprice']
    
    # Optimized approach: Iterate through columns and apply grouped transform for interpolation
    # followed by forward fill. 'transform' maintains the DataFrame's original index,
    # making it more efficient than 'apply' for this type of operation.
    for col in cols_to_fill:
        # Interpolate missing values within each item_id group, looking only forward.
        # Then, ffill any remaining NaNs (especially leading ones or those after interpolation)
        processed_priced_data[col] = processed_priced_data.groupby('item_id')[col].transform(
            lambda x: x.interpolate(method=interp_method, limit_direction='forward').ffill()
        )

    # Final global ffill: A safety net for any remaining NaNs that might occur at the very beginning
    # of the dataset where a grouped ffill might not have a preceding value.
    processed_priced_data.ffill(inplace=True)

    print(f"\nNaNs after grouped forward-only interpolation and forward fills: \n{processed_priced_data.isnull().sum()}")
    print(f"Total NaNs in processed_priced_data (may be non-zero for leading NaNs): {processed_priced_data.isnull().sum().sum()}")
    
    # Reorder columns to ensure consistent output structure, especially for saving and future reading
    final_columns_order = ['timestamp', 'item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'totalvol', 'wprice']
    processed_priced_data = processed_priced_data[final_columns_order]
    
    return processed_priced_data









def data_preprocess2(
    read:               bool = True, 
    filepath:           str = "../data", 
    read_path:          str = "../data/processed_data.csv", 
    write:              bool = False, 
    interp_method:      str = 'linear', 
    filter_volume:      bool = True,
    filter_threshold:   int = 0.99
) -> pd.DataFrame:
   
    if read:
        try:
            df = pd.read_csv(
                f'{read_path}',
                names=[
                    'item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice',
                    'lowPriceVolume', 'timestamp', 'totalvol', 'wprice'
                ]
            )
            print(f"Successfully loaded processed data from {read_path}")
            return df
        except FileNotFoundError:
            print(f"Processed data not found at {read_path}. Proceeding with raw data processing.")
            pass # Fall through to process raw data if file not found
        except Exception as e:
            # Catch other potential errors during file reading (e.g., parsing issues)
            print(f"Error reading processed data from {read_path}: {e}. Proceeding with raw data processing.")
            pass

    # --- Load and Initial Clean Raw Data ---
    raw_pricedata = pd.read_csv(f'{filepath}/data.csv', names=['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp'])
    
    # Sorting is crucial for correct interpolation and forward-filling
    raw_pricedata = raw_pricedata.sort_values(by=['item_id', 'timestamp']).reset_index(drop=True)

    initial_rows = len(raw_pricedata)
    # Remove duplicate (item_id, timestamp) entries, keeping the last observed value
    raw_pricedata.drop_duplicates(subset=['item_id', 'timestamp'], keep='last', inplace=True)
    if len(raw_pricedata) < initial_rows:
        print(f"Removed {initial_rows - len(raw_pricedata)} duplicate (item_id, timestamp) entries from raw data.")
    
    # Load data properties to get series_length (assuming it's on the second line)
    try:
        if filter_volume:
            with open(f'{filepath}/data_properties.txt', "r") as file:
                lines = file.readlines()
        else:
            with open(f'{filepath}/data_properties_full.txt', "r") as file:
                lines = file.readlines()
        series_length = int(lines[1].replace("\n", ""))
    except FileNotFoundError:
        print(f"Error: data_properties.txt not found at {filepath}. Cannot determine series_length.")
        # Fallback or raise error if series_length is critical and cannot be determined
        series_length = raw_pricedata['timestamp'].nunique() # Fallback to number of unique timestamps
        print(f"Using unique timestamp count as series_length: {series_length}")
    except IndexError:
        print(f"Error: data_properties.txt at {filepath} does not contain expected series_length on the second line.")
        series_length = raw_pricedata['timestamp'].nunique()
        print(f"Using unique timestamp count as series_length: {series_length}")

    print(f"Raw data loaded. Initial NaNs (should be 0 for raw inputs, unless explicit in CSV):\n{raw_pricedata.isnull().sum()}")

    group_raw_pricedata = raw_pricedata.groupby('item_id').nunique()

    if filter_volume:
        # Filter out items with insufficient data points
        filtered_indexes = group_raw_pricedata[group_raw_pricedata['timestamp'] < filter_threshold * series_length].index
        raw_pricedata = raw_pricedata[~raw_pricedata['item_id'].isin(filtered_indexes)]
    print(f"\nAfter filtering: {len(raw_pricedata['item_id'].unique())} item(s) remaining.")
    
    # --- Step 1: Create a COMPLETE GRID of (timestamp, item_id) combinations ---
    all_timestamps = np.sort(raw_pricedata['timestamp'].unique()) # Use np.sort for efficiency
    all_item_ids = raw_pricedata['item_id'].unique()

    full_index = pd.MultiIndex.from_product([all_timestamps, all_item_ids], names=['timestamp', 'item_id'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Merge the raw data onto the full grid, introducing NaNs for missing combinations
    processed_priced_data = pd.merge(full_df, raw_pricedata, on=['timestamp', 'item_id'], how='left')

    print(f"\nNaNs after creating full grid and left merging (expected for initially missing combos):\n{processed_priced_data.isnull().sum()}")

    # Fill NA volumes with 0 prior to calculation to avoid NaN propagation
    processed_priced_data['highPriceVolume'] = processed_priced_data['highPriceVolume'].fillna(0)
    processed_priced_data['lowPriceVolume'] = processed_priced_data['lowPriceVolume'].fillna(0)
    
    processed_priced_data['totalvol'] = processed_priced_data['highPriceVolume'] + processed_priced_data['lowPriceVolume']

    # Temporarily fill avgHighPrice/avgLowPrice for 'wprice' calculation if they are NaN at this stage.
    # We use grouped forward fill (ffill) to prevent data leakage and ensure that
    # price calculations only use information from the past.
    processed_priced_data['avgHighPrice_calc'] = processed_priced_data.groupby('item_id')['avgHighPrice'].transform(lambda x: x.ffill())
    processed_priced_data['avgLowPrice_calc'] = processed_priced_data.groupby('item_id')['avgLowPrice'].transform(lambda x: x.ffill())

    # Calculate 'wprice' (weighted price).
    # Use np.where to handle division by zero safely: if 'totalvol' is 0, 'wprice' is NaN.
    # Otherwise, calculate 'wprice' based on volumes and average prices.
    processed_priced_data['wprice'] = np.where( # Use np.where
        processed_priced_data['totalvol'] == 0,
        np.nan, # Use np.nan for explicit missing values
        (processed_priced_data['highPriceVolume'] / processed_priced_data['totalvol']) * \
        (processed_priced_data['avgHighPrice_calc'] - processed_priced_data['avgLowPrice_calc']) + \
        processed_priced_data['avgLowPrice_calc']
    )
    
    # Drop the temporary calculation columns
    processed_priced_data.drop(columns=['avgHighPrice_calc', 'avgLowPrice_calc'], inplace=True)

    # --- Step 3: Grouped Interpolation (forward-only) and Forward Fill for all relevant columns ---
    # This is CRITICAL for time-series data to prevent future information (data leakage)
    # from influencing current or past predictions.
    print("\nStarting grouped forward-only interpolation and forward fills...")
    
    cols_to_fill = ['avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'totalvol', 'wprice']
    
    # Optimized approach: Iterate through columns and apply grouped transform for interpolation
    # followed by forward fill. 'transform' maintains the DataFrame's original index,
    # making it more efficient than 'apply' for this type of operation.
    for col in cols_to_fill:
        # Interpolate missing values within each item_id group, looking only forward.
        # Then, ffill any remaining NaNs (especially leading ones or those after interpolation)
        processed_priced_data[col] = processed_priced_data.groupby('item_id')[col].transform(
            lambda x: x.interpolate(method=interp_method, limit_direction='forward').ffill()
        )

    # Final global ffill: A safety net for any remaining NaNs that might occur at the very beginning
    # of the dataset where a grouped ffill might not have a preceding value.
    processed_priced_data.ffill(inplace=True)

    print(f"\nNaNs after grouped forward-only interpolation and forward fills: \n{processed_priced_data.isnull().sum()}")
    print(f"Total NaNs in processed_priced_data (may be non-zero for leading NaNs): {processed_priced_data.isnull().sum().sum()}")
    
    # Reorder columns to ensure consistent output structure, especially for saving and future reading
    final_columns_order = ['timestamp', 'item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'totalvol', 'wprice']
    processed_priced_data = processed_priced_data[final_columns_order]

    # Saving output if 'write' is True
    if write:
        processed_priced_data.to_csv(f'{filepath}/processed_data.csv', mode='w', header=False, index=False)
        print(f"Processed data saved to {filepath}/processed_data.csv")
    
    return processed_priced_data

def data_preprocess(read: bool, filepath: str = "../data", read_path: str = "../data/processed_data.csv", write: bool = False, interp_method: str = 'linear') -> pd.DataFrame:
    ### Read has higher priority than write
    if read:
        df = pd.read_csv(
            f'{read_path}',
            names=[
                'item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice',
                'lowPriceVolume', 'timestamp', 'totalvol', 'wprice'
            ]
        )
        return df

    #Load the data
    raw_pricedata = pd.read_csv(f'{filepath}/data.csv', names=['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp'])
    #Load the data properties
    with open(f'{filepath}/data_properties.txt', "r") as file:
        lines = file.readlines()
        file.close()
    series_length = int(lines[1].replace("\n", ""))

    #Keep constantly only constantly traded items
    group_raw_pricedata = raw_pricedata.groupby('item_id').nunique()
    filtered_indexes = group_raw_pricedata[group_raw_pricedata['timestamp'] < 0.99*series_length].index #filter out lower volume items
    raw_pricedata = raw_pricedata[~raw_pricedata['item_id'].isin(filtered_indexes)]
    
    print(f"\nAfter filtering: {len(raw_pricedata['item_id'].unique())} item(s) remaining.")

    # interpolate missing values
    processed_priced_data = raw_pricedata.interpolate(method=interp_method)
    
    #Weighted average of High/Low Price by High/Low Volume
    processed_priced_data['totalvol'] = processed_priced_data['highPriceVolume'] + processed_priced_data['lowPriceVolume']
    processed_priced_data['wprice'] = (processed_priced_data['highPriceVolume']/processed_priced_data['totalvol']) * (processed_priced_data['avgHighPrice'] - processed_priced_data['avgLowPrice']) + processed_priced_data['avgLowPrice']

    #Saving output
    if write:
        processed_priced_data.to_csv(f'{filepath}/processed_data.csv', mode='w', header=False, index=False)
    
    return processed_priced_data

def alchemy_preprocess(read: bool, filepath: str = "../data", read_path: str = "../data/alchemy_data.csv", write: bool = False) -> pd.DataFrame:
    ### Read has higher priority than write
    if read:
        df = pd.read_csv(f'{read_path}', names=['item', 'price'], index_col=0)
        return df

    with open(f'{filepath}/namealchemy.json', "r") as file:
        alc_data = json.load(file)
        file.close()
    with open(f'{filepath}/nameID.json', "r") as file:
        name_data = json.load(file)
        file.close()

    # Convert dictionary to DataFrame
    high_alchemy = pd.DataFrame(list(alc_data.items()), columns=["item", "price"])
    nameID = pd.DataFrame(list(name_data.items()), columns=["item", "item_id"])

    #Processing tables
    reference = pd.merge(nameID, high_alchemy, on="item", how="inner")
    reference = reference.drop([0,1])
    reference.set_index('item_id', inplace=True)

    #Saving output
    if write:
        reference.to_csv(f'{filepath}/alchemy_data.csv', mode='w', header=False, index=True)

    return reference

def volatility_market(market_data: pd.DataFrame, smoothing: int = 20) -> pd.Series:
    price_market_data = market_data.pivot(index="timestamp", columns="item_id", values="wprice")
    #Aggregate volatility
    volatilityitems = price_market_data.rolling(window=smoothing).std()
    volatilitymarket = volatilityitems.sum(axis=1)
    #Scaling
    corr_price_market = price_market_data.corr()
    volatilitymarket = volatilitymarket/corr_price_market.shape[1]

    return pd.DataFrame(volatilitymarket, columns=['market_vix'])

if __name__ == "__main__":
    test = data_preprocess2(read=False, write=True) #run in interactive

