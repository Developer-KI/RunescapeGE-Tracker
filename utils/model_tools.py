import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import  utils.api_fetcher as fetcher
import  scipy.stats as stats
import  os
from    matplotlib.ticker import MaxNLocator, ScalarFormatter
from    utils.data_pipeline import alchemy_preprocess
from    sklearn.metrics import mean_absolute_error
from    matplotlib.gridspec import GridSpec
from    statsmodels.tsa.api import SimpleExpSmoothing
from    scipy.stats import norm, kurtosis, skew, shapiro, jarque_bera
from    typing import TypeVar, cast, Dict
from    datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json

print(os.getcwd())
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

with open('../data/nameID.json', 'r') as file:
    name_to_id: Dict[str, int] = json.load(file)

# Create the inverted dictionary for reverse lookups (also run only once)
id_to_name: Dict[int, str] = {value: key for key, value in name_to_id.items()}

# --- Step 2: Create a Fast Lookup Function ---

def item_name(query: int|str) -> str|int:
    """
    Performs a fast, bi-directional lookup using pre-built dictionaries.
    """
    if isinstance(query, int):
        if query in id_to_name:
            return id_to_name.get(query)
        else: raise ValueError(f"ID '{query}' not found.")
    elif isinstance(query, str):
        if query in name_to_id:
            return name_to_id.get(query)
        else: raise ValueError(f"Name '{query}' not found.")
    else: raise ValueError("Input must be integer ID or string name")






















def target_time_features(y:pd.DataFrame, feature_col:str, time_feature:int = 2) -> pd.DataFrame:
    data = y.copy()
    lagged_data = {f'lag{t}': data[feature_col].shift(t) for t in range(1, time_feature + 1)}
    lagged_df = pd.DataFrame(lagged_data, index=data.index)
    
    return pd.concat([data, lagged_df], axis=1)

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

def target_rolling_features(y:pd.DataFrame, feature_col:str, window:int = 2) -> pd.DataFrame:
    data = y.copy()
    data['rolling_mean'] = data[feature_col].rolling(window).mean()
    data['rolling_std'] = data[feature_col].rolling(window).std()
    return data[['rolling_mean', 'rolling_std']]

def mase(y_true:np.ndarray,y_pred:np.ndarray,y_train:np.ndarray,m:int=1) -> float:
    naive_errors = np.abs(y_train[m:]-y_train[:-m])
    naive_mae = (naive_errors).mean()
    # Model forecast error
    mae_model = np.abs(y_true - y_pred).mean()
    if naive_mae==0:
        print("WARNING: Naive MAE=0, numerical instability due to epsilon division.")
    if len(y_train) <= m:
    # Handle cases where y_train is too short for differencing
        print(f"WARNING: y_train length ({len(y_train)}) is too short for period m={m}.")
    return mae_model/np.maximum(naive_mae, np.finfo(np.float64).eps)

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
        raise ValueError("holdout period must be non-negative.")
    
    full_x = data.drop(target_col, axis=1)
    full_y = data[target_col]
    
    return full_x, full_y

def score_tree_model(full_y, holdout, y_holdout, final_preds_holdout, final_train_Y) -> tuple:

    final_holdout_mase = mase(y_holdout, final_preds_holdout, final_train_Y, 1)
    final_holdout_mae = mean_absolute_error(y_holdout, final_preds_holdout)

    # Calculate Final Holdout DA
    # Get the last actual price from the training data before the holdout
    last_train_actual = full_y.iloc[-holdout - 1] 
    
    # Combine last training actual with holdout actuals mand predictions
    combined_actuals_for_da = np.concatenate(([last_train_actual], y_holdout))
    combined_preds_for_da = np.concatenate(([last_train_actual], final_preds_holdout))
    
    final_holdout_da = calculate_directional_accuracy(combined_actuals_for_da, combined_preds_for_da)

    return final_holdout_mae, final_holdout_mase, final_holdout_da
def outlier_detection(best_model, full_y, full_x, window_size, outlier_threshold) -> pd.Series:
#outlier detection REWORK 
        window_size = 20
        epsilon = 1e-8
        residuals = full_y - best_model.predict(full_x.to_numpy(dtype='float32'))
        mad_resid = residuals.rolling(window=window_size).apply(lambda x: np.median(abs(x - np.median(x))))
        mod_z_resid = 0.6745 * (residuals - residuals.rolling(window=window_size).median()) / (mad_resid + epsilon) #hard-coded value why?
        outliers = full_y[abs(mod_z_resid) > outlier_threshold]

        return outliers

def beta(
    data:           pd.DataFrame,
    item:           int,
    market_index:   pd.Series,
    start:          str|None = None,
    end:            str|None = None,
) -> float: 
    timestamp_slice = None
    market_index_slice = None

    market_index_slice = market_index.loc[start:end]
    
    beta = market_index_slice.corr(data[item]) * (data[item].loc[start:end].std()/market_index.loc[start:end].std())
    return beta


def generate_dates_with_rollover(start_date_str, day_numbers_str):
    """
    Generates a list of dates by conditionally rolling back the month.
    The month rolls back by one only when a day number in the list is
    greater than the last day number processed.

    Args:
        start_date_str (str): The date to start from, in 'YYYY-MM-DD' format.
        day_numbers_str (str): A comma-separated string of day numbers to use.
                               e.g., "27, 25, 4, 1, 25, 16".
    Returns:
        pd.DatetimeIndex: A pandas DatetimeIndex containing the generated dates.
    """
    try:
        current_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        day_numbers = [int(d.strip()) for d in day_numbers_str.split(',')]
    except (ValueError, IndexError):
        print("Error: Please check your date and day number formats.")
        return pd.DatetimeIndex([])

    generated_dates = []
    
    # Initialize the last_day_seen to a high value to handle the first entry
    last_day_seen = current_date.day + 1

    for day in day_numbers:
        # Check for a month rollover condition
        # This happens if the new day number is greater than the last one seen
        if day > last_day_seen:
            current_date -= relativedelta(months=1)
        
        # Create a new date object with the current month/year and the new day
        new_date = current_date.replace(day=day)
        generated_dates.append(new_date)
        
        # Update the last day seen for the next iteration
        last_day_seen = day

    # Convert the list of datetime objects into a DatetimeIndex
    return pd.to_datetime(generated_dates)

def create_item_index(data: pd.DataFrame|list, items:list, type:str, base_value:int=100) -> pd.Series:
    """
    Args:
        data (pd.DataFrame|str): Ensure passing in a list of price and volume DataFrames is ordered [price,volume].
        items (list): A comma separated list of item IDs.
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