import  pandas as pd
import  numpy as np

def rolling_median_average_deviation(best_model, full_y, full_x, window_size, outlier_threshold) -> pd.Series:
    window_size = 20
    epsilon = 1e-8
    residuals = full_y - best_model.predict(full_x.to_numpy(dtype='float32'))

    rolling_median = residuals.rolling(window=window_size).median()
    mad_resid = (residuals - rolling_median).abs().rolling(window=window_size).median()
    mod_z_resid = 0.6745 * (residuals - residuals.rolling(window=window_size).median()) / (mad_resid + epsilon) #scaling so similar to gaussian
    outliers = full_y[abs(mod_z_resid) > outlier_threshold]
    return outliers

def rolling_zscore(full_y, window, outlier_threshold):
    demeaned_y = full_y-full_y.rolling(window=window).mean()
    zscore = demeaned_y/full_y.rolling(window=window).std()
    outliers = full_y[abs(zscore) > outlier_threshold]
    return outliers

def iqr(full_y, outlier_threshold):
    lower_percentile = np.percentile(full_y, 25)
    upper_percentile = np.percentile(full_y, 75)
    iqr = upper_percentile-lower_percentile
    outlier_filter = (full_y < lower_percentile-outlier_threshold*iqr) | (full_y > outlier_threshold*iqr)
    outliers = full_y[outlier_filter]
    return outliers

def ewm_z_residuals(
    full_y: pd.Series,
    smoothing: int,
    lower_bound: float|None = 3,
    upper_bound: float|None = 3,
    iterative: bool = False,
    g: int = 0
) -> pd.Series:
    """
    Detects outliers based on a Z-score calculated on EWMA residuals. 

    Parameters:
    - full_y (pd.Series): The data to analyze.
    - smoothing (int): The span for the EWMA calculation.
    - zscore_threshold (float): The absolute Z-score threshold for outlier detection.

    Returns:
    - pd.Series: The data points identified as outliers.
    """
    ewm_mean = full_y.ewm(span=smoothing).mean()
    residuals = full_y - ewm_mean
    ewm_std_residuals = residuals.ewm(span=smoothing).std() #outlier inflation
    # A small constant is added to the denominator to prevent division by zero.
    zscores = residuals / (ewm_std_residuals + 1e-9)
    outlier_mask = pd.Series(False, index=full_y.index)

    # Check for upper outliers
    upper_outliers = zscores > upper_bound
    outlier_mask = outlier_mask | upper_outliers

    # Check for lower outliers
    lower_outliers = zscores < -lower_bound
    outlier_mask = outlier_mask | lower_outliers
    
    filtered = full_y[outlier_mask]
    if iterative:
        print("Additional Passes:", g+1)
        if g == 2:
            print(outlier_mask)
        print(f"Removed: {outlier_mask.sum()}")
        if g+1 == 20 or not outlier_mask.any():
            return filtered[outlier_mask]
        
        ewm_z_residuals(full_y.drop(filtered.index), smoothing, lower_bound, upper_bound,True, g+1)
        
    return full_y[outlier_mask]  

import pandas as pd
import numpy as np

def ewm_z_residuals2(
    full_y: pd.Series,
    smoothing: int,
    lower_bound: float | None = 3,
    upper_bound: float | None = 3,
    iterative: bool = False,
    g: int = 0
) -> pd.Series:
    """
    Detects outliers based on a Z-score calculated on EWMA residuals.
    
    Parameters:
    - full_y (pd.Series): The data to analyze.
    - smoothing (int): The span for the EWMA calculation.
    - upper_bound (float): The Z-score threshold for upper outliers.
    - lower_bound (float): The Z-score threshold for lower outliers.
    - iterative (bool): If True, performs multiple passes to find more outliers.
    - g (int): Internal recursion counter.
    
    Returns:
    - pd.Series: The data points identified as outliers.
    """
    # Base case: Stop recursion after 20 passes or if no outliers were found
    if iterative and (g >= 20 or not full_y.any()):
        return pd.Series(dtype=full_y.dtype)

    ewm_mean = full_y.ewm(span=smoothing).mean()
    residuals = full_y - ewm_mean
    ewm_std_residuals = residuals.ewm(span=smoothing).std()
    std_safe = ewm_std_residuals.replace(0, np.nan)
    zscores = residuals / std_safe
    
    upper_outliers = zscores > upper_bound
    lower_outliers = zscores < -lower_bound
    outlier_mask = upper_outliers | lower_outliers
    
    # Check if any new outliers were found
    if iterative and outlier_mask.any():
        print(f"Additional Passes: {g + 1}")
        print(f"Found {outlier_mask.sum()} new outliers.")
        
        # Get the outliers from the current pass
        current_outliers = full_y[outlier_mask]
        
        # Get the remaining "clean" data for the next pass
        non_outlier_data = full_y[~outlier_mask]
        
        # Recurse on the non-outlier data and combine the results
        subsequent_outliers = ewm_z_residuals2(
            non_outlier_data, 
            smoothing, 
            lower_bound, 
            upper_bound, 
            iterative=True, 
            g=g + 1
        )
        
        return pd.concat([current_outliers, subsequent_outliers])
    
    return full_y[outlier_mask]


    

#TODO implement into RFTS/XGB
def rolling_volume(full_y: pd.Series, volume: pd.Series, window: int, threshold: float):

    log_return = np.log(full_y / full_y.shift(1))

    ratio = np.abs(log_return) / volume

    rolling_mean_ratio = ratio.rolling(window=window).mean()
    rolling_std_ratio = ratio.rolling(window=window).std()

    outlier_threshold = rolling_mean_ratio + rolling_std_ratio * threshold

    is_outlier = ratio > outlier_threshold

    return full_y[is_outlier]

