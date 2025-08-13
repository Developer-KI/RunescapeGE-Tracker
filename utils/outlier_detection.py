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

def ewm(
    full_y: pd.Series,
    smoothing: int,
    lower_bound: float|None = None,
    upper_bound: float|None = None
) -> pd.Series:
    """
    Detects outliers based on a scaled Exponentially Weighted Moving Average (EWMA).
    
    Parameters:
    - full_y (pd.Series): The data to analyze.
    - smoothing (int): The span for the EWMA calculation.
    - upper_bound (float): Multiplicative scaling factor above given series.
    - lower_bound (float): Multiplicative scaling factor below given series.
    
    Returns:
    - pd.Series: The data points identified as outliers.
    """
    ewm_series = full_y.ewm(span=smoothing).mean()
    
    # Initialize a mask with all False values
    outlier_mask = pd.Series(False, index=full_y.index)
    
    if upper_bound is not None:
        upper_outliers_mask = full_y > ewm_series * upper_bound
        outlier_mask = outlier_mask | upper_outliers_mask
    if lower_bound is not None:
        lower_outliers_mask = full_y < ewm_series * lower_bound
        outlier_mask = outlier_mask | lower_outliers_mask
        
    return full_y[outlier_mask]
