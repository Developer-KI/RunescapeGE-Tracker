import  os
import sys
# Get the directory of the current script (which is 'utils')
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the JSON file
json_path = os.path.join(current_dir, '..', 'data', 'nameID.json')

import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import  scipy.stats as stats
from    sklearn.metrics import mean_absolute_error
from    typing import TypeVar, cast, Dict
from    datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json

# plt.rcParams.update({
#     'axes.facecolor': '#2E2E2E',
#     'axes.titlecolor': 'white',
#     'figure.facecolor': '#1E1E1E',
#     'axes.labelcolor': 'white',
#     'xtick.color': 'white',
#     'ytick.color': 'white',
#     'grid.color': '#444444',
#     'axes.edgecolor': 'white'
# })
plt.rcParams.update({
    "figure.facecolor": "#000000",  
    "axes.facecolor": "#000000",   
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "gray",
    "legend.facecolor": "#303030",
    "legend.labelcolor": "#A1A1A1",
    "axes.titlecolor": "#A1A1A1"

})

with open(json_path, 'r') as file:
    name_to_id: Dict[str, int] = json.load(file)

# Create the inverted dictionary for reverse lookups (also run only once)
id_to_name: Dict[int, str] = {value: key for key, value in name_to_id.items()}

# --- Step 2: Create a Fast Lookup Function ---

def item_name(query: int|np.integer|str) -> str|int:
    """
    Performs a fast, bi-directional lookup using pre-built dictionaries.
    """
    if isinstance(query, (int, np.integer)):
        # Use .get() with an f-string for concise error handling
        result = id_to_name.get(query)
        if result is None:
            raise ValueError(f"ID '{query}' not found.")
        return result
    
    elif isinstance(query, str):
        result = name_to_id.get(query)
        if result is None:
            raise ValueError(f"Name '{query}' not found.")
        return result
    else:
        raise ValueError("Input must be integer ID or string name")

def create_feature_lags(data:pd.DataFrame, feature_col:str, lags:list[int]) -> pd.DataFrame:
    working_data = data.copy()
    lagged_data = {f'lag{t}': working_data[feature_col].shift(t) for t in lags}
    lagged_df = pd.DataFrame(lagged_data, index=working_data.index)
    result_df = pd.concat([working_data, lagged_df], axis=1)
    
    return result_df

def target_rolling_features(y:pd.DataFrame, feature_col:str, window:int = 2) -> pd.DataFrame:
    data = y.copy()
    data['rolling_mean'] = data[feature_col].rolling(window).mean().shift(1)
    data['rolling_std'] = data[feature_col].rolling(window).std().shift(1)
    return data

def rolling_threshold_classification(features:pd.DataFrame, window:int, diffpercent:float) -> pd.DataFrame: 
    if window==0:
        raise ValueError("Window size must be larger than 0.")
    elif window==1:
        print("Window size 1, proceeding with no window.")
    
    newfeatures= features.copy()
    
    for i in range(0, features.shape[0], window):
        end = min(i + window, features.shape[0])  # Handle end boundary
        rolling_mean = newfeatures.iloc[i:end, :].mean()  # Compute mean for the chunk
        newfeatures.iloc[i:end, :] = rolling_mean  # Assign the rolling mean to the original DataFrame
        maskhigh = newfeatures.iloc[i:end] > features.iloc[i]*(1+diffpercent/100) #thresholds
        masklow = newfeatures.iloc[i:end] < features.iloc[i]*(1-diffpercent/100)
        newfeatures[maskhigh]=2
        newfeatures[masklow]=0
        newfeatures[~masklow & ~maskhigh]=1
        
    return newfeatures.astype(int)

def convert_numpy(X) -> np.ndarray:
    """Convert pandas DataFrame to NumPy array if necessary."""
    if  isinstance(X, pd.DataFrame):
        return X.to_numpy()
    else:
        print("convert_numpy Conversion failure.")
        return X  

DataFrameOrSeries = TypeVar('DataFrameOrSeries', pd.Series, pd.DataFrame)
def ensure_datetime_index(s:DataFrameOrSeries) -> DataFrameOrSeries: 
    # Check if the index is already a DatetimeIndex
    if isinstance(s.index, pd.DatetimeIndex):
        return s.copy()

    # Create the new DatetimeIndex
    new_index = pd.to_datetime(s.index, unit='s', errors='raise', utc=True)
    
    # Handle DataFrame and Series differently
    if isinstance(s, pd.DataFrame):
        # For a DataFrame, create a new DataFrame with the new index
        s = s.set_index(new_index)
    elif isinstance(s, pd.Series):
        # For a Series, directly assign the new index
        s.index = new_index
        
    return s.copy()

def spread_rolling_z(feature1:pd.Series, feature2:pd.Series, window:int) -> pd.Series:
    
    spread = feature1-feature2
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    rolling_z = (spread - rolling_mean)/rolling_std
    return rolling_z #contains NaNs!

def calculate_directional_accuracy(actual_prices: np.ndarray, predicted_prices: np.ndarray) -> float:
   
    # Ensure arrays are of compatible length for comparison
    if len(actual_prices) != len(predicted_prices):
        raise ValueError("actual_prices and predicted_prices must have the same length.")

    # We need at least two points to determine a direction (current vs. previous)
    if len(actual_prices) < 2:
        return np.nan # Or raise an error, as DA is not meaningful for < 2 points

    actual_changes = np.diff(actual_prices)
    predicted_changes_from_last_actual = predicted_prices[1:] - actual_prices[:-1]

    # Option 2: Direction of actual price relative to *previous actual price*
    actual_changes_from_last_actual = actual_prices[1:] - actual_prices[:-1]

    # Determine the signs of changes
    actual_direction = np.sign(actual_changes_from_last_actual)
    predicted_direction = np.sign(predicted_changes_from_last_actual)

    # Handle cases where actual_change is zero (no change) - typically counted as incorrect
    # unless you explicitly want to count 'no change' as a third category.
    # For simplicity, we'll count exact zero as wrong if the prediction isn't also zero.

    correct_predictions = np.sum(actual_direction == predicted_direction)

    return (correct_predictions / len(actual_changes_from_last_actual)) * 100

def prep_tree_model(data:pd.DataFrame, target_col:str, holdout:int) -> tuple:

    if holdout<=0:
        raise ValueError("Holdout period must be non-negative.")
    
    holdout_x = data.drop(target_col, axis=1).iloc[-holdout:]
    holdout_y = data[target_col].iloc[-holdout:]
    train_x = data.drop(target_col, axis=1).iloc[:-holdout]
    train_y = data[target_col].iloc[:-holdout]

    
    return train_x, train_y, holdout_x, holdout_y

def score_tree_model(
        y_train:        pd.Series,
        y_holdout:      pd.Series,
        holdout_preds:  pd.Series, 
        ) -> tuple:

    holdout_mase = mase(y_holdout, y_train, holdout_preds, 1)
    holdout_mae = mean_absolute_error(y_holdout, holdout_preds)

    # Calculate Holdout DA
    # Get the last actual price from the training data before the holdout
    last_train_actual = y_train.iloc[-1]
    
    # Combine last training actual with holdout actuals mand predictions
    combined_actuals_for_da = np.concatenate(([last_train_actual], y_holdout))
    combined_preds_for_da = np.concatenate(([last_train_actual], holdout_preds))
    
    holdout_da = calculate_directional_accuracy(combined_actuals_for_da, combined_preds_for_da)

    return holdout_mae, holdout_mase, holdout_da

def mase(y_true:    pd.Series|np.ndarray, 
         y_train:   pd.Series|np.ndarray, 
         y_pred:    pd.Series|np.ndarray, 
         m:         int=1
        ) -> float:
    
    naive_errors = np.abs(y_train[m:].values-y_train[:-m].values)
    naive_mae = (naive_errors).mean()
    # Model forecast error
    mae_model = np.abs(y_true - y_pred).mean()
    if naive_mae==0:
        print("WARNING: Naive MAE=0, numerical instability due to epsilon division.")
    if len(y_train) <= m:
    # Handle cases where y_train is too short for differencing
        print(f"WARNING: y_train length ({len(y_train)}) is too short for period m={m}.")
    return mae_model/np.maximum(naive_mae, np.finfo(np.float64).eps)

def calculate_returns(price_series: pd.Series|pd.DataFrame, return_periods: str|None = None) -> pd.Series|pd.DataFrame:
    """
    Calculates returns for a given price series, handling both base returns and resampling.
    """
    #induces an aggregation distortion

    ratio = price_series/price_series.shift(1).dropna()
    raw_return = ratio -1

    if return_periods is None or return_periods.lower() == '5m':
        return raw_return
    
    elif isinstance(return_periods, str):
        try:
            compound_returns = (ratio).resample(return_periods).prod()
            return compound_returns - 1
        except Exception as e:
            raise ValueError(f"Invalid resample rule '{return_periods}'. Error: {e}")
    else:
        raise ValueError("return_periods must be a valid resample string or None.")

def beta(
    price_data: pd.DataFrame,
    item: int|str,
    market_index: pd.Series,
    return_periods: str|None = None,
    start: str|None = None,
    end: str|None = None,
) -> float:
    """
    Calculates the Beta of an item relative to a market index.
    """
    if isinstance(item, str):
        item = item_name(item)

    combined_prices = pd.DataFrame({
        'item': price_data[item],
        'market': market_index,
    }).loc[start:end]

    item_returns = calculate_returns(combined_prices['item'], return_periods=return_periods)
    market_returns = calculate_returns(combined_prices['market'], return_periods=return_periods)
    
    covariance = item_returns.cov(market_returns)
    market_variance = market_returns.var()

    if market_variance == 0:
        return np.nan
    
    beta_value = covariance / market_variance
    return beta_value

def create_item_index(data: pd.DataFrame|list, items:list[int], type:str, base_value:int=100) -> pd.Series:
    """
    Args:
        data (pd.DataFrame|str): Ensure passing in a list of price and volume DataFrames is ordered [price,volume].
        items (list): A comma separated list of item IDs.
        type (string): Equal-weighted or volume-weighted indexing
        base_value (int, default 100): Index starting value.
    """
    base_value = 100
    if isinstance(data, pd.DataFrame):
        data_columns = set(data.columns)
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], pd.DataFrame) and isinstance(data[1], pd.DataFrame):
        data_columns = set(data[0].columns)
    else:
        raise ValueError("Invalid data type for check.")

    if not all(column in data_columns for column in items):
        missing_items = [column for column in items if column not in data_columns]
        print(f"Warning: One or more items in the 'items' list do not exist in the DataFrame: {missing_items}")
        print("Removing missing items and proceeding...")
        items_to_keep = [column for column in items if column in data_columns]
        items = items_to_keep
    
    if type == 'equal':
        if data is list:
            raise ValueError("data should be a single DataFrame")
        price_matrix = data
        selection_price = price_matrix[items]
        base_prices = selection_price.iloc[0,:]

        price_ratio = data.div(base_prices, axis=1)
        mean_ratio = price_ratio.mean(axis=1) 
        index_equal_weight = mean_ratio * base_value
        return index_equal_weight
    elif type == 'vprice':
        if data is pd.DataFrame:
            raise ValueError("vprice indicies must include a list of price and volume matrix DataFrames")
        price_matrix = data[0]
        selection_price = price_matrix[items]
        volume_matrix = data[1]
        selection_volume = volume_matrix[items]
        base_prices = selection_price.iloc[0,:]
        price_ratio = price_matrix.div(base_prices, axis=1)

        vprice_matrix = selection_price * selection_volume
        sum_vprice_matrix = vprice_matrix.sum(axis=1)
        vprice_ratio = sum_vprice_matrix/sum_vprice_matrix.iloc[0]
        index_volume_weight = base_value * vprice_ratio
        return index_volume_weight
    else: raise ValueError("Select valid index type")

def rsi(price_data: pd.Series, periods: int) -> pd.Series:
    
    delta = price_data.diff(1)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    avg_gain_smoothed = gains.ewm(com=periods - 1, adjust=False).mean()
    avg_loss_smoothed = losses.ewm(com=periods - 1, adjust=False).mean()
    # 4. Initialize an RSI Series with NaNs
    rsi = pd.Series(np.nan, index=price_data.index)
    
    # 5. Handle the division by zero case explicitly
    # Only calculate RS where avg_loss is not zero
    mask = avg_loss_smoothed != 0
    rs = pd.Series(np.nan, index=price_data.index)
    rs[mask] = avg_gain_smoothed[mask] / avg_loss_smoothed[mask]
    
    # 6. Calculate RSI
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    # 7. Set RSI to 100 where avg_loss is 0
    rsi[avg_loss_smoothed == 0] = 100
    
    return rsi

def hurst(data: pd.Series) -> float:
    for n in range(10, 100, 10):
        # Skip if n is not an integer divisor of the data length.
        # This is not strictly necessary but ensures clean chunks.
        if 1000 % n != 0:
            continue
        
        # Divide the series into n chunks
        chunks = np.array_split(data, 1000 // n)

        # Calculate the mean and standard deviation for each chunk
        rs_n = []
        for chunk in chunks:
            mean_chunk = chunk.mean()
            std_chunk = chunk.std()
            
            # Calculate cumulative deviations from the mean
            cumulative_deviation = np.cumsum(chunk - mean_chunk)
            
            # Calculate the rescaled range
            # Use a small value to avoid division by zero for flat chunks
            if std_chunk != 0:
                rs_n.append((np.max(cumulative_deviation) - np.min(cumulative_deviation)) / std_chunk)

        if rs_n:
            rs_values.append(np.mean(rs_n))
            n_values.append(n)

    # Now, you have the n_values and their corresponding average R/S values.
    # The Hurst exponent is the slope of the line that fits the
    # log-log plot of these values.
    log_rs = np.log(rs_values)
    log_n = np.log(n_values)

    # Use polyfit to find the slope (Hurst exponent)
    hurst_exponent = np.polyfit(log_n, log_rs, 1)[0]
    return hurst_exponent

def volatility_market(price_data: pd.DataFrame, aggregation: str|None = None, smoothing: int = 20) -> pd.Series:
    #Aggregate volatility
    volatility_items = calculate_returns(price_data, aggregation)
    volatility_items = price_data.rolling(window=smoothing).std().dropna()
    volatility_sum = volatility_items.sum(axis=1)
    #Scaling
    volatility_market = volatility_sum/price_data.shape[1]
    volatility_market.name = 'market_vix'

    return volatility_market