import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import DataPipeline as pipeline
import APIFetcher as fetcher
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.api import SimpleExpSmoothing

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
def item_name(id):
    if id==13190:
        return 'Bond'
    else: return str(read.loc[read[0]==int(id)][1].item())

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
    naive_mae = (naive_errors).mean()
    # Model forecast error
    mae_model = np.abs(y_true - y_pred).mean()
    if naive_mae==0:
        print("WARNING: Naive MAE=0, numerical instability due to epsilon division.")
    if len(y_train) <= m:
    # Handle cases where y_train is too short for differencing
        print(f"WARNING: y_train length ({len(y_train)}) is too short for period m={m}.")
    return mae_model/np.maximum(naive_mae, np.finfo(np.float64).eps)

def ensure_numpy(X):
    """Convert pandas DataFrame to NumPy array if necessary."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return X  

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
 
def plot_pred_vs_price(data: pd.DataFrame, model, holdout_pred:np.array, lookback: int = 0, std_factor: float = 1.96):
    Y = data[data.columns[0]].to_numpy()
    item = data.columns[0] 
    # Preserve original timestamps for plotting
    time_index = data.index.to_numpy()  

    # adj_index now represents the full time range for plotting, covering the 'lookback' period
    adj_index = time_index[-lookback:]
    
    # Ensure lookback is at least as large as holdout, otherwise adjust.
    if lookback < len(holdout_pred):
        print(f"Warning: 'lookback' ({lookback}) is less than 'holdout' ({len(holdout_pred)}). Adjusting lookback to holdout value for plotting consistency.")
        lookback = len(holdout_pred)
        adj_index = time_index[-lookback:]
        
    # Determine the slice points for training/test and holdout periods
    training_test_plot_end_idx = lookback - len(holdout_pred) 
    
    # Corresponding time indices for the plot
    training_test_time_indices = adj_index[:training_test_plot_end_idx]
    holdout_time_indices = adj_index[training_test_plot_end_idx:]

    # Confidence Interval Calculation
    pred_std = np.std(holdout_pred)  
    upper_bound = holdout_pred + std_factor * pred_std
    lower_bound = holdout_pred - std_factor * pred_std

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

        # --- Plotting Setup ---

        # Keep constrained_layout=True for automatic spacing
        fig = plt.figure(figsize=(12, 10), constrained_layout=True) 
        
        # Use GridSpec with height_ratios
        gs = GridSpec(3, 1, figure=fig, height_ratios=[4, 2, 1], hspace=0.05) 

        # **Row 1: Main Subplot - Predictions vs. Actual**
        ax_main = fig.add_subplot(gs[0, 0]) 
        ax_main.plot(adj_index, Y[-lookback:], marker="o", markersize=2, linestyle="-", label="Actual", color='white')
        ax_main.plot(holdout_time_indices, holdout_pred, marker="o", markersize=2, linestyle="-", label="Predicted", color="red")
        ax_main.fill_between(holdout_time_indices, lower_bound, upper_bound, color="#5DD4FF39", alpha=0.3,
                                label=f"{(stats.norm.cdf(std_factor) - stats.norm.cdf(-std_factor)) * 100:.2f}% Confidence Interval (Gaussian)")

        ax_main.set_ylabel("GP Price")
        ax_main.set_title(f"{item_name(item)} [{item}] Predicted vs. Actual Price")
        ax_main.legend()
        ax_main.grid()


        # **Row 2: Residuals Subplot (Full Width, with color change)**
        ax_residuals = fig.add_subplot(gs[1, 0], sharex=ax_main) 
        
        ax_residuals.plot(training_test_time_indices, residuals_training_test_percent, 
                        marker="o", markersize=2, linestyle="-", label="Residuals (Train/Test)", color='grey')
        
        ax_residuals.plot(holdout_time_indices, residuals_holdout_for_plot_percent, 
                        marker="o", markersize=2, linestyle="-", label="Residuals (Holdout)", color="skyblue")
        
        ax_residuals.axhline(y=0, color="black", linestyle="--", alpha=0.7)  

        ax_residuals.set_ylabel("Residuals Percentage") 
        ax_residuals.legend()
        ax_residuals.grid()


        # **Row 3: Histogram of Residuals (Separate Plot)**
        ax_hist = fig.add_subplot(gs[2, 0]) 
        ax_hist.hist(residuals_holdout_for_plot, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax_hist.set_xlabel("Residual Value")
        ax_hist.set_ylabel("Frequency") 
        ax_hist.grid(axis='y', linestyle='--', alpha=0.7)


        # --- General Formatting ---
        plt.setp(ax_main.get_xticklabels(), visible=False) 
        
        # Directly set rotation for the x-tick labels on the residuals subplot
        # This is the bottom-most plot with visible x-axis labels.
        for label in ax_residuals.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right') # Adjust horizontal alignment after rotation

        # REMOVED: fig.autofmt_xdate(rotation=45) 

        plt.show()
    else: 
        # Residual Calculation
        residuals = Y[holdout_time_indices] - holdout_pred #not sure if leaking
        residuals_percent= (residuals/Y[holdout_time_indices])*100
        
        fig = plt.figure(figsize=(12, 10), constrained_layout=True) 
        
        # Use GridSpec with height_ratios
        gs = GridSpec(3, 1, figure=fig, height_ratios=[4, 2, 1], hspace=0.05) 

        # **Row 1: Main Subplot - Predictions vs. Actual**
        ax_main = fig.add_subplot(gs[0, 0]) 
        ax_main.plot(adj_index, Y[-lookback:], marker="o", markersize=2, linestyle="-", label="Actual", color='white')
        ax_main.plot(holdout_time_indices, holdout_pred, marker="o", markersize=2, linestyle="-", label="Predicted", color="red")
        ax_main.fill_between(holdout_time_indices, lower_bound, upper_bound, color="#5DD4FF39", alpha=0.3,
                                label=f"{(stats.norm.cdf(std_factor) - stats.norm.cdf(-std_factor)) * 100:.2f}% Confidence Interval (Gaussian)")

        ax_main.set_ylabel("GP Price")
        ax_main.set_title(f"{item_name(item)} [{item}] Predicted vs. Actual Price")
        ax_main.legend()
        ax_main.grid()


        # **Row 2: Residuals Subplot (Full Width, with color change)**
        ax_residuals = fig.add_subplot(gs[1, 0], sharex=ax_main) 
        
        ax_residuals.plot(holdout_time_indices, residuals_percent, 
                        marker="o", markersize=2, linestyle="-", label="Residuals (Holdout)", color="skyblue")
        
        ax_residuals.axhline(y=0, color="black", linestyle="--", alpha=0.7)  

        ax_residuals.set_ylabel("Residuals Percentage") 
        ax_residuals.legend()
        ax_residuals.grid()


        # **Row 3: Histogram of Residuals (Separate Plot)**
        ax_hist = fig.add_subplot(gs[2, 0]) 
        ax_hist.hist(residuals_holdout_for_plot, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax_hist.set_xlabel("Residual Value")
        ax_hist.set_ylabel("Frequency") 
        ax_hist.grid(axis='y', linestyle='--', alpha=0.7)


        # --- General Formatting ---
        plt.setp(ax_main.get_xticklabels(), visible=False) 
        
        # Directly set rotation for the x-tick labels on the residuals subplot
        # This is the bottom-most plot with visible x-axis labels.
        for label in ax_residuals.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right') # Adjust horizontal alignment after rotation

        # REMOVED: fig.autofmt_xdate(rotation=45) 



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