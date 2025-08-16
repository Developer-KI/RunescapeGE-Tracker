import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import  utils.api_fetcher as fetcher
import  scipy.stats as stats
import  os
from    matplotlib.ticker import MaxNLocator, ScalarFormatter
import  utils.model_tools as tools
from    utils.data_pipeline import alchemy_preprocess, data_preprocess2 
from    sklearn.metrics import mean_absolute_error
from    matplotlib.gridspec import GridSpec
from    statsmodels.tsa.api import SimpleExpSmoothing
from    scipy.stats import norm, kurtosis, skew, shapiro, jarque_bera
import  pytz
from    typing import cast
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

_DEFAULT_TITLE = object()

def daytime_shade(
        feature_for_index:  pd.Series, 
        time_start:         str = '11:00:00', 
        time_end:           str = '20:00:00', 
        plot_type:          plt.Axes|None = None
) -> None:
    
    dates = feature_for_index.index.date
    unique_dates = pd.to_datetime(pd.Series(dates).unique())
    
    for date in unique_dates:
        # Define the day period for first day (e.g., 11 AM to 8 PM)
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
    y_slice = y.loc[start:end]

    if x is None and y is None:
        raise ValueError("Input x or y data")
    elif x is not None and y is None:
        raise ValueError("x arguments require a y argument")
    
    y_slice = tools.ensure_datetime_index(y).loc[start:end]

    if x is not None:
        x_slice = tools.ensure_datetime_index(x).loc[start:end]
        ax.plot(x_slice, y_slice)
        ax.set_xlabel(xlab)
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
    plt.xticks(rotation=45)
    if show:
        plt.show()
    return fig, ax


def plot_recent_alch_vs_price(item_id: int) -> None:
    reference = alchemy_preprocess(read=True)

    if item_id in reference.index:
        df = data_preprocess2()
        df = df.pivot(index="timestamp", columns="item_id", values="wprice")[item_id]
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(df.index, unit='s'), df.values, marker="o", markersize='2', linestyle="-", label=f"{reference.loc[item_id,'item']} Price")
        plt.axhline(y=reference.loc[item_id, 'price'], color='cyan', linestyle='-', label='High Alchemy Price')

        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Recent Alchemy vs Realized Price")
        plt.xticks(rotation=45)  # Rotate timestamps for clarity
        plt.legend()
        plt.grid()

        plt.show()
    else: 
        raise ValueError("Invalid ID")
    
def plot_historical_alch_vs_price(item_id: int) -> None:
    reference = alchemy_preprocess(read=True)

    if item_id in reference.index:
        df = fetcher.fetch_historical(item_id)
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(df['timestamp'], unit='s'), df['price'], marker="o", markersize='2', linestyle="-", label=f"{reference.loc[item_id,'item']} Price")
        plt.axhline(y=reference.loc[item_id, 'price'], color='cyan', linestyle='-', label='High Alchemy Price')

        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Historical Alchemy vs Realized Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid()

        plt.show()
    else: 
        raise Exception("Invalid ID")

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
    X = data.drop(data.columns[0], axis=1).to_numpy()
    Y = data[data.columns[0]].to_numpy()
    adj_index = data.index[lookback:]

    Y_pred = model.predict(X[lookback:])
    residuals = Y[lookback:] - Y_pred

    plt.figure(figsize=(12, 6))
    plt.plot(adj_index, residuals, marker="o", markersize=2, linestyle="-", label="Residuals", color='green')
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.7)  # Reference line
    ax=plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=25)) 
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Time")
    plt.ylabel("Error (Residuals)")
    plt.title("Residual Errors Over Time")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()
    return residuals

def plot_pred_vs_price(data: pd.DataFrame, model, holdout_pred:np.ndarray, lookback: int = 0, fill_outliers=None, std_factor: float = 1.96):
    if fill_outliers is not None:
        Y = data[data.columns[0]].copy()
        Y.loc[fill_outliers.index] = np.nan
        Y=Y.ffill().to_numpy()
    else:
            Y = data[data.columns[0]].to_numpy()
    item = int(data.columns[0])
    # Preserve original timestamps for plotting
    time_index = data.index.to_numpy()  
    adj_index = time_index[-lookback:]
    
    if lookback > data.shape[0]:
        print(f"Warning: 'lookback' ({lookback}) is greater than dataset size ({data.shape[0]}). Adjusting lookback to max size for plotting consistency.")
        lookback = data.shape[0]
        adj_index = time_index
    if lookback < len(holdout_pred):
        print(f"Warning: 'lookback' ({lookback}) is less than 'holdout' ({len(holdout_pred)}). Adjusting lookback to holdout value for plotting consistency.")
        lookback = len(holdout_pred)
        adj_index = time_index[-lookback:]
        
    # Determine the slice points for training/test and holdout periods
    training_test_plot_end_idx = lookback - len(holdout_pred) 
    
    training_test_time_indices = adj_index[:training_test_plot_end_idx]
    holdout_time_indices = adj_index[training_test_plot_end_idx:]

    # Confidence Interval Calculation
    pred_std = np.std(holdout_pred)  
    upper_bound = holdout_pred + std_factor * pred_std
    lower_bound = holdout_pred - std_factor * pred_std
    
    fig = plt.figure(figsize=(12, 10), constrained_layout=True) 

    if not isinstance(model, SimpleExpSmoothing):
        X = data.drop(data.columns[0], axis=1).to_numpy()

        # Model predictions 
        #Keep in mind that this generates predictions for all folds from a final trained model
        #which is inherently leaking data *between* folds, as opposed to generating predictions from within each fold.
        #The leaking approach is good to examine the training fit given the maximum data, but cannot
        #help decide hyperparameters or compare performance between folds
        Y_pred_full_lookback = model.predict(X[-lookback:]) 

        # Residual Calculation
        residuals_full_lookback = Y[-lookback:] - Y_pred_full_lookback #not sure if leaking
        residuals_full_percent= (residuals_full_lookback/Y[-lookback:])*100
        
        # Separate residuals into training/test and holdout for different colors
        residuals_training_test = residuals_full_lookback[:training_test_plot_end_idx]
        residuals_holdout_for_plot = residuals_full_lookback[training_test_plot_end_idx:]
        residuals_training_test_percent = residuals_full_percent[:training_test_plot_end_idx]
        residuals_holdout_for_plot_percent = residuals_full_percent[training_test_plot_end_idx:]
        
        # Use GridSpec with height_ratios
        gs = GridSpec(3, 1, figure=fig, height_ratios=[4, 2, 1], hspace=0.05) 

        # **Row 1: Main Subplot - Predictions vs. Actual**
        ax_main = fig.add_subplot(gs[0, 0]) 
        ax_main.plot(adj_index, Y[-lookback:], marker="o", markersize=2, linestyle="-", label="Actual", color='white')
        ax_main.plot(holdout_time_indices, holdout_pred, marker="o", markersize=2, linestyle="-", label="Predicted", color="red")
        ax_main.fill_between(holdout_time_indices, lower_bound, upper_bound, color="#5DD4FF39", alpha=0.3,
                                label=f"{(stats.norm.cdf(std_factor) - stats.norm.cdf(-std_factor)) * 100:.2f}% Confidence Interval (Gaussian)")
        # Only plot outliers that are within the current plotting window
        if fill_outliers is not None and not fill_outliers.empty:
            # adj_index is your x-axis (timestamps), fill_outliers.index should be timestamps too
            outlier_idx = np.intersect1d(adj_index, fill_outliers.index)
            ax_main.plot(
                outlier_idx,
                fill_outliers.loc[outlier_idx],
                'X', color='yellow', markersize=8, markeredgecolor='black', label='Detected Outliers'
            )
        # **Row 2: Residuals Subplot (Full Width, with color change)**
        ax_residuals = fig.add_subplot(gs[1, 0], sharex=ax_main) 
        ax_residuals.plot(training_test_time_indices, residuals_training_test_percent, 
                        marker="o", markersize=2, linestyle="-", label="Residuals (Train/Test)", color='grey')
        ax_residuals.plot(holdout_time_indices, residuals_holdout_for_plot_percent, 
                        marker="o", markersize=2, linestyle="-", label="Residuals (Holdout)", color="skyblue")
        
        # **Row 3: Histogram of Residuals (Separate Plot)**
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
        shapiro_stat, shapiro_p_value = shapiro(residuals_holdout_for_plot)
        
        textstr = '\n'.join((
            r'$\mu$: %.4f' % (mu_norm,),
            r'$\sigma$: %.4f' % (std_norm,),
            r'Skewness: %.3f' % (skew(residuals_holdout_for_plot),),
            r'Kurtosis: %.3f' % (kurtosis(residuals_holdout_for_plot),),
            r'Jaque-Bera: %.3f (p=%.3f)' % (jb_stat,jb_p_value),
            r'Shapiro-Wilk: %.3f (p=%.3f)' % (shapiro_stat,shapiro_p_value)
        ))
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6)
        ax_hist.text(0.02, 0.98, textstr, transform=ax_hist.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left', bbox=props)

    else: 
        # Residual Calculation
        residuals = Y[-len(holdout_pred):] - holdout_pred #not sure if leaking
        residuals_percent= (residuals/Y[-len(holdout_pred):])*100
        
        # Use GridSpec with height_ratios
        gs = GridSpec(2, 2, figure=fig, height_ratios=[4, 2], width_ratios=[1,3], hspace=0.05) 

        # **Row 1: Main Subplot - Predictions vs. Actual**
        ax_main = fig.add_subplot(gs[0, :]) 
        ax_main.plot(adj_index, Y[-lookback:], marker="o", markersize=2, linestyle="-", label="Actual", color='white')
        ax_main.plot(holdout_time_indices, holdout_pred, marker="o", markersize=2, linestyle="-", label="Predicted", color="red")
        ax_main.fill_between(holdout_time_indices, lower_bound, upper_bound, color="#5DD4FF39", alpha=0.3,
                                label=f"{(stats.norm.cdf(std_factor) - stats.norm.cdf(-std_factor)) * 100:.2f}% Confidence Interval (Gaussian)")

        # **Row 2: Residuals Subplot (Full Width, with color change)**
        ax_residuals = fig.add_subplot(gs[1, 1]) 
        ax_residuals.plot(holdout_time_indices, residuals_percent, 
                        marker="o", markersize=2, linestyle="-", label="Residuals (Holdout)", color="skyblue")
        ax_residuals.set_xlim(holdout_time_indices[0], holdout_time_indices[-1])
    
        # **Row 3: Histogram of Residuals (Separate Plot)**
        ax_hist = fig.add_subplot(gs[1, 0]) 
        ax_hist.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

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

    #General Formatting
    plt.setp(ax_main.get_xticklabels(), visible=False) 
    
    # Directly set rotation for the x-tick labels on the residuals subplot
    # This is the bottom-most plot with visible x-axis labels.
    for label in ax_residuals.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right') # Adjust horizontal alignment after rotation

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

        # Predict on train/test sets
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Store errors (Mean Absolute Error or RMSE)
        train_errors.append(loss(y_train, train_preds))
        test_errors.append(loss(y_test, test_preds))


    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(values, train_errors, label="Train Error", marker='o')
    plt.plot(values, test_errors, label="Test Error", marker='o')
    plt.xlabel("Max Depth")
    plt.ylabel("Error (MAE)")
    plt.legend()
    plt.title(f"Train vs. Test Error ({param})")
    plt.show()