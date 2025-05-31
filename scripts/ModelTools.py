import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import DataPipeline as pipeline
import APIFetcher as fetcher

def target_time_features(y: pd.DataFrame, feature_col: str, time_feature: int = 2) -> pd.DataFrame:
    data = y.copy()
    for t in range(1, time_feature + 1):
        data[f'lag{t}'] = data[feature_col].shift(t)
    return data

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

def plot_pred_vs_price(Y_test: pd.Series, X_test: pd.DataFrame, model) -> None:
    Y_pred = pd.Series(model.predict(X_test))

    plt.figure(figsize=(10, 5))
    plt.plot(Y_test.index, Y_test.values, marker="o", markersize='2', linestyle="-", label='Realized')
    plt.plot(Y_test.index, Y_pred.values, marker="o", markersize='1', linestyle="-", label='Predicted')
    ax=plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.xaxis.get_major_formatter().set_scientific(False) 
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=25)) 
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Predicted vs Realized Price")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()

    plt.show()

def plot_classification_vs_price(hist_pricedata,hidden_states,item, model):
    timescale = hist_pricedata.index

    state_colors = {0: "red", 1: "gray", 2: "green"}
    fig, ax = plt.subplots()

    for t in range(1,len(timescale)-1):
        ax.axvspan(timescale[t], timescale[t + 1], color=state_colors[hidden_states[t]], alpha=0.07)

    ax.plot(timescale, hist_pricedata[item], label="Price Data")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.ticklabel_format(useOffset=False) 

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid()

    plt.show()

