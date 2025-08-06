import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import  utils.api_fetcher as fetcher
import  scipy.stats as stats
import  os
from    matplotlib.ticker import MaxNLocator, ScalarFormatter
from    utils.data_pipeline import alchemy_preprocess, data_preprocess 
from    sklearn.metrics import mean_absolute_error
from    matplotlib.gridspec import GridSpec
from    statsmodels.tsa.api import SimpleExpSmoothing
from    scipy.stats import norm, kurtosis, skew, shapiro, jarque_bera
from    typing import TypeVar, cast
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

read = pd.read_csv('../data/alchemy_data.csv', header=None)
##
def item_name(id:int) -> str:
    if id==13190:
        return 'Bond'
    else: return str(read.loc[read[0]==int(id)][1].item())

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
    #type checker isnt smart enough
    if not isinstance(s.index, pd.DatetimeIndex):
        return cast(DataFrameOrSeries, s.set_index(pd.to_datetime(s.index, unit='s', errors='raise', utc=True))) #explicit cast, type checker isn't happy
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