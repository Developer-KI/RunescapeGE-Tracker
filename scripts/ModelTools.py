import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import DataPipeline as pipeline
import APIFetcher as fetcher
import scipy.stats as stats

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
    "legend.labelcolor": "#A1A1A1"
})
def target_time_features(y: pd.DataFrame, feature_col: str, time_feature: int = 2) -> pd.DataFrame:
    data = y.copy()
    lagged_data = {f'lag{t}': data[feature_col].shift(t) for t in range(1, time_feature + 1)}
    lagged_df = pd.DataFrame(lagged_data, index=data.index)
    
    return pd.concat([data, lagged_df], axis=1)

def rolling_threshold_classification(features:pd.DataFrame, window:int, diffpercent: float): 
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

def target_rolling_features(y: pd.DataFrame, feature_col: str, window: int = 2) -> pd.DataFrame:
    data = y.copy()
    data['rolling_mean'] = data[feature_col].rolling(window).mean()
    data['rolling_std'] = data[feature_col].rolling(window).std()
    return data[['rolling_mean', 'rolling_std']]

def plot_recent_alch_vs_price(item_id: int) -> None:
    reference = pipeline.alchemy_preprocess(read=True)

    if item_id in reference.index:
        df = pipeline.data_preprocess(read=True)
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
        raise Exception("Invalid ID")

def plot_historical_alch_vs_price(item_id: int) -> None:
    reference = pipeline.alchemy_preprocess(read=True)

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
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.ticklabel_format(useOffset=False) 

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.grid()

    plt.show()

def plot_residuals(data: pd.DataFrame, model, lookback: int = 0):
    X = data.drop(data.columns[0], axis=1).to_numpy()
    Y = data[data.columns[0]].to_numpy()
    adj_index = data.index[lookback:]

    Y_pred = model.predict(X[lookback:])
    residuals = Y[lookback:] - Y_pred

    plt.figure(figsize=(12, 6))
    plt.plot(adj_index, residuals, marker="o", markersize=2, linestyle="-", label="Residuals", color='green')
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.7)  # Reference line
    ax=plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.xaxis.get_major_formatter().set_scientific(False) 
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=25)) 
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Time")
    plt.ylabel("Error (Residuals)")
    plt.title("Residual Errors Over Time")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()

def mase(y_true:np.ndarray,y_pred:np.ndarray,y_train:np.ndarray,m:int=1) -> float:
    naive_errors = np.abs(y_train[m:]-y_train[:-m])
    naive_mae = sum(naive_errors)/(len(naive_errors)-m)
    naive_mae = (naive_errors).mean()
    # Model forecast error
    mae_model = np.abs(y_true - y_pred).mean()
    if naive_mae==0:
        print("WARNING: Naive MAE=0, numerical instability due to epsilon division.")
    return mae_model/np.maximum(naive_mae, np.finfo(np.float64).eps)

def ensure_numpy(X):
    """Convert pandas DataFrame to NumPy array if necessary."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return X  

def plot_pred_vs_price(data: pd.DataFrame, model, best_index: np.array = None, lookback: int = 0, std_factor: float = 1.96):
    item = data.columns[0]
    
    # Preserve original timestamps for plotting
    time_index = data.index.to_numpy()  
    adj_index = time_index[lookback:]  

    # Convert data to NumPy for efficiency
    X = data.drop(data.columns[0], axis=1).to_numpy()
    Y = data[data.columns[0]].to_numpy()

    # Model predictions (NumPy-based)
    Y_pred = model.predict(X[lookback:])

    # Confidence Interval Calculation
    pred_std = np.std(Y_pred)  # Simplified uncertainty measure
    upper_bound = Y_pred + std_factor * pred_std
    lower_bound = Y_pred - std_factor * pred_std

    # Residual Calculation
    residuals = Y[lookback:] - Y_pred  

    # Create figure with subplots for predictions + residuals
    fig, ax1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # **Top Subplot: Predictions vs. Actual**
    ax1[0].plot(adj_index, Y[lookback:], marker="o", markersize=2, linestyle="-", label="Actual", color='white')
    ax1[0].plot(adj_index, Y_pred, marker="o", markersize=2, linestyle="-", label="Predicted", color='red')
    ax1[0].fill_between(adj_index, lower_bound, upper_bound, color="#829FFF10", alpha=0.3,
                        label=f"{(stats.norm.cdf(std_factor) - stats.norm.cdf(-std_factor)) * 100:.2f}% Confidence Interval (Gaussian)")

    ax1[0].set_ylabel("GP Price")
    ax1[0].set_title(f"Item {item} Predicted vs. Actual Price")
    ax1[0].legend()
    ax1[0].grid()

    # **Bottom Subplot: Residuals**
    ax1[1].plot(adj_index, residuals, marker="o", markersize=2, linestyle="-", label="Residuals", color='green')
    ax1[1].axhline(y=0, color="black", linestyle="--", alpha=0.7)  # Reference line for zero-error

    ax1[1].set_ylabel("Residuals")
    ax1[1].legend()
    ax1[1].grid()

    # Formatting the x-axis for both subplots
    ax1[1].xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax1[1].xaxis.get_major_formatter().set_scientific(False)
    ax1[1].xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))  

    plt.xticks(adj_index[::len(adj_index)//10], rotation=45)  # Keep timestamps, but reduce clutter
    plt.tight_layout()
    plt.show()