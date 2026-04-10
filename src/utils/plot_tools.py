import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import  src.data_ingestion.data_fetcher as fetcher
import  scipy.stats as stats
from    matplotlib.ticker import MaxNLocator, ScalarFormatter
import  src.utils.model_tools as tools
from    src.data_processing.data_pipeline import alchemy_preprocess, data_preprocess2 
from    sklearn.metrics import mean_absolute_error
from    matplotlib.gridspec import GridSpec
from    statsmodels.tsa.api import SimpleExpSmoothing
from    scipy.stats import norm, kurtosis, skew, jarque_bera
import  pytz
from    typing import cast
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

_DEFAULT_TITLE = object()

def daytime_shade(
        feature_for_index:  pd.Series|pd.DataFrame|pd.DatetimeIndex, 
        time_start:         str = '11:00:00', 
        time_end:           str = '20:00:00', 
        plot_type:          plt.Axes|None = None
) -> None:
    if isinstance(feature_for_index, pd.Series) or isinstance(feature_for_index, pd.DataFrame):
        dates = feature_for_index.index.date
        unique_dates = pd.to_datetime(pd.Series(dates).unique())
    elif isinstance(feature_for_index, pd.DatetimeIndex):
        unique_dates = feature_for_index.unique()
    else: raise ValueError("Shaded feature must be datetimeIndex or Series and DataFrame containing one.")
    
    for date in unique_dates:
        day_start = pd.Timestamp(f"{date} {time_start}")
        day_end = pd.Timestamp(f"{date} {time_end}")
        if plot_type is not None:
           plot_type.axvspan(day_start, day_end, color='gray', alpha=0.2, zorder=0)
        else:
            plt.axvspan(day_start, day_end, color='gray', alpha=0.2, zorder=0)

def plot_features(
    y:      pd.Series | None = None,
    x:      pd.Series | None = None,
    start:  str | None = None,
    end:    str | None = None,
    xlab:   str = "X",
    ylab:   str = "Y",
    title:  str|object = _DEFAULT_TITLE,
    show:   bool = False #currently useless for interactive mode/Jupyter type environments
) -> tuple[plt.Figure, plt.Axes]: # Explicitly return the plot objects
    
    fig, ax = plt.subplots(figsize=(10, 5))

    if x is None and y is None:
        raise ValueError("Input x or y data")
    elif x is not None and y is None:
        raise ValueError("x arguments require a y argument")
    
    y_slice = tools.ensure_datetime_index(y).loc[start:end]

    if x is not None:
        x_slice = tools.ensure_datetime_index(x).loc[start:end]
        ax.plot(x_slice, y_slice)
        ax.set_xlabel(xlab)
        ax.ticklabel_format(style='plain', useOffset=False)
        title_to_use: str = "X against Y" if title is _DEFAULT_TITLE else cast(str, title)
    else:
        ax.plot(y_slice)
        title_to_use: str = "Y Over Time" if title is _DEFAULT_TITLE else cast(str, title)
        daytime_shade(y_slice)
    
    ax.set_title(title_to_use)
    ax.set_ylabel(ylab)
    plt.xticks(rotation=45)
    ax.grid()
    if show:
        plt.show()

    return fig, ax

def plot_price(
    item:  int|str,
    price_data: pd.DataFrame|pd.Series,
    start:  str | None = None,
    end:    str | None = None,
    marker: bool = False,
    show:   bool = False #currently useless for interactive mode/Jupyter type environments
) -> tuple[plt.Figure, plt.Axes]: # Explicitly return the plot objects
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if isinstance(item, str):
        item = tools.item_name(item)

    if isinstance(price_data.columns, pd.MultiIndex):
        high_price = price_data[('avgLowPrice', item)]
        low_price = price_data[('avgHighPrice', item)]
        y_slice_high = tools.ensure_datetime_index(high_price).loc[start:end]
        y_slice_low = tools.ensure_datetime_index(low_price).loc[start:end]
        daytime_shade(y_slice_high)
        if marker:
            ax.plot(y_slice_high, label='High Price', marker='.', markerfacecolor='yellow')
            ax.plot(y_slice_low, label="Low Price", marker='.', markerfacecolor='yellow')
        else:
            ax.plot(y_slice_high, label='High Price')
            ax.plot(y_slice_low, label="Low Price")
        plt.legend()

    else:
        wprice = price_data[item]  
        y_slice = tools.ensure_datetime_index(wprice).loc[start:end]
        daytime_shade(y_slice)
        ax.plot(y_slice, marker='.', markerfacecolor='yellow')

    ax.set_title(fr'{tools.item_name(item)} ({item})')
    ax.set_ylabel("GP")
    ax.grid()
    plt.xticks(rotation=45)

    if show:
        plt.show()

    return fig, ax


def plot_item_market_divergence(
    data:           pd.DataFrame,
    item:           int,
    market_index:   pd.Series,
    return_periods: str|None = None,
    start:          str|None = None,
    end:            str|None = None,
    show:           bool = False
) -> tuple[plt.Figure, plt.Axes]: 

    fig, ax = plt.subplots(figsize=(10,5))
    
    market_index = tools.ensure_datetime_index(market_index)
    market_index_slice = market_index.loc[start:end]
    timestamp_slice = market_index.index[start:end]
    if return_periods is None or return_periods == '5m' :
        market_returns = market_index_slice.pct_change().dropna()
        item_returns = data[item].pct_change().dropna()
    elif return_periods is not None and isinstance(return_periods, str): 
        market_returns = tools.calculate_returns(market_index_slice, return_periods=return_periods)
        item_returns = tools.calculate_returns(data[item].loc[timestamp_slice], return_periods=return_periods)
    else: raise ValueError('Valid return period required')
    
    return_diff = market_returns-item_returns
    daytime_shade(market_index_slice)
    ax.plot(return_diff)
    ax.set_title(fr"$\mathbf{{{tools.item_name(item)}}}$ [{item}] Percent Divergence Against Index") #add dynamic index name (based on var name)
    
    ax.set_ylabel("% Difference")
    ax.grid()
    plt.xticks(rotation=45)
    if show:
        plt.show()
    return fig, ax

def plot_feature_divergence(
    feature1:   pd.Series,
    feature2:   pd.Series,
    type:       str,
    start:      str|None = None,
    end:        str|None = None,
    window:     int|None = None,
    show:       bool = False
) -> tuple[plt.Figure, plt.Axes]: # Explicitly return the plot objects
    
    fig, ax = plt.subplots(figsize=(10,5))
    
    feature1 = tools.ensure_datetime_index(feature1)
    feature2 = tools.ensure_datetime_index(feature2)

    feature1_slice = feature1.loc[start:end]
    feature2_slice = feature2.loc[start:end]

    if type == 'percent':
        feature1_pct = feature1_slice.pct_change()
        feature2_pct = feature2_slice.pct_change()
        feature_pct_diff = feature1_pct-feature2_pct
        ax.plot(feature_pct_diff)
        ax.set_title("Percent Divergence")
        ax.set_ylabel("% Difference")
    elif type == 'raw':
        ax.plot(feature1_slice-feature2_slice)
        ax.set_title("Raw Divergence")
        ax.set_ylabel("Difference")
    elif type == 'z':
        if window is None:
            raise ValueError("Select rolling window size")
        rolling_z = tools.spread_rolling_z(feature1_slice, feature2_slice, window)
        ax.plot(rolling_z)
        ax.set_title("Standardized Divergence")
        ax.set_ylabel("z-score")
    else: raise ValueError('Select a valid divergence type')
    daytime_shade(feature1_slice)
    ax.grid()
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.xticks(rotation=45)
    if show:
        plt.show()
    return fig, ax


def plot_recent_alch_vs_price(item_id: int, show: bool = False) -> tuple[plt.Figure, plt.Axes]:
    reference = alchemy_preprocess(read=True)

    if item_id in reference.index:
        fig, ax = plt.subplots(figsize=(10, 5))
        df = data_preprocess2()
        df = df.pivot(index="timestamp", columns="item_id", values="wprice")[item_id]
        ax.plot(pd.to_datetime(df.index, unit='s'), df.values, marker="o", markersize='2', linestyle="-", label=f"{reference.loc[item_id,'item']} Price")
        ax.axhline(y=reference.loc[item_id, 'price'], color='cyan', linestyle='-', label='High Alchemy Price')

        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.set_title("Recent Alchemy vs Realized Price")
        plt.xticks(rotation=45)
        ax.legend()
        ax.grid()

        if show:
            plt.show()
        return fig, ax
    else:
        raise ValueError("Invalid ID")
    
def plot_historical_alch_vs_price(item_id: int, show: bool = False) -> tuple[plt.Figure, plt.Axes]:
    reference = alchemy_preprocess(read=True)

    if item_id in reference.index:
        fig, ax = plt.subplots(figsize=(10, 5))
        df = fetcher.fetch_historical(item_id)
        ax.plot(pd.to_datetime(df['timestamp'], unit='s'), df['price'], marker="o", markersize='2', linestyle="-", label=f"{reference.loc[item_id,'item']} Price")
        ax.axhline(y=reference.loc[item_id, 'price'], color='cyan', linestyle='-', label='High Alchemy Price')

        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.set_title("Historical Alchemy vs Realized Price")
        ax.legend()
        plt.xticks(rotation=45)
        ax.grid()

        if show:
            plt.show()
        return fig, ax
    else:
        raise Exception("Invalid ID")

def plot_acf(
    series:     pd.Series,
    lags:       range | list | None = None,
    alpha:      float = 0.05,
    title:      str | object = _DEFAULT_TITLE,
    show:       bool = False
) -> tuple[plt.Figure, plt.Axes]:

    series = series.dropna()
    n = len(series)
    mean = series.mean()

    if lags is None:
        lags = range(40)
    lags = list(lags)

    acf_values = []
    denom = np.sum((series - mean) ** 2)
    for lag in lags:
        if lag == 0:
            acf_values.append(1.0)
        else:
            numer = np.sum((series.values[lag:] - mean) * (series.values[:-lag] - mean))
            acf_values.append(numer / denom)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lags, acf_values, width=0.3, color='skyblue', edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='white', linewidth=0.8)

    z = norm.ppf(1 - alpha / 2)
    ci = z / np.sqrt(n)
    ax.axhline(ci, color='red', linestyle='--', linewidth=1, label=f'{(1 - alpha)*100:.0f}% CI')
    ax.axhline(-ci, color='red', linestyle='--', linewidth=1)
    ax.fill_between(lags, -ci, ci, color='red', alpha=0.08)

    title_to_use: str = "Autocorrelation Function (ACF)" if title is _DEFAULT_TITLE else cast(str, title)
    ax.set_title(title_to_use)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.legend()
    if show:
        plt.show()

    return fig, ax


def plot_classification_vs_price(hist_pricedata,hidden_states,item, model):
    timescale = hist_pricedata.index

    state_colors = {0: "red", 1: "gray", 2: "green"}
    fig, ax = plt.subplots()

    for t in range(1,len(timescale)-1):
        ax.axvspan(timescale[t], timescale[t + 1], color=state_colors[hidden_states[t]], alpha=0.07)

    ax.plot(timescale, hist_pricedata[item])
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax.ticklabel_format(useOffset=False) 

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.grid()

    plt.show()

def plot_residuals(data: pd.DataFrame, model, lookback: int = 0) -> np.ndarray:
    X = data.drop(data.columns[0], axis=1)
    Y = data[data.columns[0]]
    adj_index = data.index[lookback:]

    Y_pred = model.predict(X.iloc[lookback:])
    residuals = Y.iloc[lookback:] - Y_pred

    plt.figure(figsize=(12, 6))
    plt.plot(adj_index, residuals, marker="o", markersize=2, linestyle="-", label="Residuals", color='green')
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.7)  # Reference line
    ax=plt.gca()
    plt.xlabel("Time")
    plt.ylabel("Error (Residuals)")
    plt.title("Residual Errors Over Time")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()
    return residuals


def plot_pred_vs_price(data: pd.DataFrame, predictions: np.ndarray, holdout_pred_n:int, lookback: int = 0, fill_outliers=None, std_factor: float = 1.96):
    if fill_outliers is not None:
        Y = data[data.columns[0]].copy()
        Y.loc[fill_outliers] = np.nan
        Y=Y.ffill().to_numpy()
    else:
            Y = data[data.columns[0]].to_numpy()
    item = int(data.columns[0])
    time_index = data.index.to_numpy()  
    adj_index = time_index[-lookback:]
    
    if lookback > data.shape[0]:
        print(f"Warning: 'lookback' ({lookback}) is greater than dataset size ({data.shape[0]}). Adjusting lookback to max size for plotting consistency.")
        lookback = data.shape[0]
        adj_index = time_index
    training_test_plot_end_idx = lookback - holdout_pred_n
    
    training_test_time_indices = adj_index[:training_test_plot_end_idx]
    holdout_time_indices = adj_index[training_test_plot_end_idx:]
    holdout_predictions = predictions[-holdout_pred_n:]

    pred_std = np.std(holdout_predictions)  
    upper_bound = holdout_predictions + std_factor * pred_std
    lower_bound = holdout_predictions - std_factor * pred_std
    
    fig = plt.figure(figsize=(12, 10), constrained_layout=True) 

    X = data.drop(data.columns[0], axis=1)

    # Model predictions 
    #Keep in mind that this generates predictions for all folds from a final trained model
    #which is inherently leaking data *between* folds, as opposed to generating predictions from within each fold.
    #The leaking approach is good to examine the training fit given the maximum data, but cannot
    #help decide hyperparameters or compare performance between folds
    Y_pred_full_lookback = predictions[-lookback:]

    residuals_full_lookback = Y[-lookback:] - Y_pred_full_lookback #not sure if leaking
    residuals_full_percent= (residuals_full_lookback/Y[-lookback:])*100
    
    residuals_training_test = residuals_full_lookback[:training_test_plot_end_idx]
    residuals_holdout_for_plot = residuals_full_lookback[training_test_plot_end_idx:]
    residuals_training_test_percent = residuals_full_percent[:training_test_plot_end_idx]
    residuals_holdout_for_plot_percent = residuals_full_percent[training_test_plot_end_idx:]
    
    gs = GridSpec(3, 1, figure=fig, height_ratios=[4, 2, 1], hspace=0.05) 

    #Main Subplot
    ax_main = fig.add_subplot(gs[0, 0]) 
    ax_main.plot(adj_index, Y[-lookback:], marker="o", markersize=2, linestyle="-", label="Actual", color='white')
    ax_main.plot(holdout_time_indices[-lookback:], holdout_predictions[-lookback:], marker="o", markersize=2, linestyle="-", label="Predicted", color="red")
    ax_main.fill_between(holdout_time_indices[-lookback:], lower_bound[-lookback:], upper_bound[-lookback:], color="#5DD4FF39", alpha=0.3,
                            label=f"{(stats.norm.cdf(std_factor) - stats.norm.cdf(-std_factor)) * 100:.2f}% Confidence Interval (Gaussian)")
    if fill_outliers:
        outlier_timestamps_arr = np.array(fill_outliers)

        visible_outlier_timestamps = np.intersect1d(adj_index, outlier_timestamps_arr)
        
        outlier_data = data.loc[visible_outlier_timestamps, data.columns[0]]
        
        ax_main.plot(
            outlier_data.index,
            outlier_data,
            'X', color='yellow', markersize=8, markeredgecolor='black', label='Detected Outliers'
        )
    #Residuals Subplot
    ax_residuals = fig.add_subplot(gs[1, 0], sharex=ax_main) 
    ax_residuals.plot(training_test_time_indices, residuals_training_test_percent, 
                    marker="o", markersize=2, linestyle="-", label="Residuals (Train/Test)", color='grey')
    ax_residuals.plot(holdout_time_indices, residuals_holdout_for_plot_percent, 
                    marker="o", markersize=2, linestyle="-", label="Residuals (Holdout)", color="skyblue")
    
    #Histogram
    ax_hist = fig.add_subplot(gs[2, 0]) 
    hist_min= np.nanpercentile(residuals_holdout_for_plot,0.5) 
    hist_max= np.nanpercentile(residuals_holdout_for_plot,99.5)
    ax_hist.hist(residuals_holdout_for_plot, bins=30, color='skyblue', edgecolor='black', alpha=0.7, range=(hist_min,hist_max), density=True)
    ax_hist.axvline(0, color='white', linestyle='--', linewidth=1)

    mu_norm, std_norm = norm.fit(residuals_holdout_for_plot)
    x_plot = np.linspace(hist_min, hist_max, 500)
    pdf_norm = norm.pdf(x_plot, mu_norm, std_norm)
    ax_hist.plot(x_plot, pdf_norm, 'r-', linewidth=1, label=r'Normal ($H_0$)')
    ax_hist.axvline(mu_norm, color='red', linestyle='-', linewidth=1)

    jb_stat, jb_p_value= jarque_bera(residuals_holdout_for_plot)
    
    textstr = '\n'.join((
        r'$\mu$: %.4f' % (mu_norm,),
        r'$\sigma$: %.4f' % (std_norm,),
        r'Skewness: %.3f' % (skew(residuals_holdout_for_plot),),
        r'Kurtosis: %.3f' % (kurtosis(residuals_holdout_for_plot),),
        r'Jaque-Bera: %.3f (p=%.3f)' % (jb_stat,jb_p_value),
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6)
    ax_hist.text(0.02, 0.98, textstr, transform=ax_hist.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    
    #Main Plot
    daytime_shade(data.iloc[-lookback:], plot_type=ax_main)
    ax_main.set_ylabel("GP Price")
    ax_main.set_title(fr"$\mathbf{{{tools.item_name(item)}}}$ [{item}] Predicted vs. Actual Price")
    ax_main.legend()
    ax_main.grid()

    # Histogram of Residuals
    ax_hist.set_xlabel("Residual Value")
    ax_hist.set_ylabel("Frequency") 
    ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
    ax_residuals.axhline(y=0, color="black", linestyle="--", alpha=0.7)  
    ax_residuals.set_ylabel("Residuals Percentage") 
    ax_residuals.legend()
    ax_residuals.grid()

    plt.setp(ax_main.get_xticklabels(), visible=False) 
    
    for label in ax_residuals.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right') 

    plt.show()

def test_train_error(data, param:str, exclude_param:dict, model_class, param_range, split=0.8,loss=mean_absolute_error):
    values = np.arange(*param_range) 
    target_item= data.columns[0]
    
    train_errors = []
    test_errors = []
    split_round=int(np.floor(split*data.shape[0]))
    X_train= data.drop(f'{target_item}',axis=1).iloc[:split_round]
    X_test= data.drop(f'{target_item}',axis=1).iloc[split_round+1:]
    y_train= data[f'{target_item}'].iloc[:split_round]
    y_test = data[f'{target_item}'].iloc[split_round+1:]
    for i in values:
        model_params = {**exclude_param, param:i}
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_errors.append(loss(y_train, train_preds))
        test_errors.append(loss(y_test, test_preds))

    plt.figure(figsize=(10, 6))
    plt.plot(values, train_errors, label="Train Error", marker='o')
    plt.plot(values, test_errors, label="Test Error", marker='o')
    plt.xlabel("Max Depth")
    plt.ylabel("Error (MAE)")
    plt.legend()
    plt.title(f"Train vs. Test Error ({param})")
    plt.show()