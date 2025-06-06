# %% Script Init
import pandas as pd
import numpy as np
import DataPipeline as pipeline
import ModelTools as tools
import Models.RFTS as myRFTS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import Models.HMM as myHMM
from sklearn.metrics import mean_absolute_error
# %% Aggregate features for model
price_data = pipeline.data_preprocess(read=True, interp_method='nearest')
price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")
vol_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="totalvol")

target_item = np.random.choice(price_matrix_items.columns)
price_items_reg = price_matrix_items[[target_item]].iloc[20:]
vol_items_reg = vol_matrix_items[[target_item]].iloc[20:]
price_items_reg.columns = [f'{target_item}']
vol_items_reg.columns = [f'{target_item}_vol']
volatility_index = pipeline.volatility_market(price_data, smoothing=20)[20:] #LEAKY
reg_data = pd.concat([price_items_reg, vol_items_reg, volatility_index], axis=1)

df_time = tools.target_time_features(reg_data, f'{target_item}', 100) 
df_roll = tools.target_rolling_features(reg_data, f'{target_item}', 20) #LEAKY
df = pd.merge(df_time, df_roll, on='timestamp', how='inner').dropna()
#%%
df_mod=df[[f'{target_item}','lag1','lag2','lag3','lag4','lag5']] #Make sure target is always the first column
# Run a RFTS model
RFTSmodel, RFTSidx = myRFTS.RFTS(df_mod, target_col=f'{target_item}',max_depth=None, n_estimators=400, min_samples_leaf=20)
tools.plot_pred_vs_price(df_mod, model=RFTSmodel,lookback=0)
#%% Run the optimized RFTS
optim, optimparam,best_test_idx = myRFTS.RFTSOptim(df_mod,target_col=f'{target_item}',n_trials=5)
#%%
tools.plot_pred_vs_price(df_mod, model=optim,best_index=best_test_idx,lookback=0)
#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
#%%
max_depths = range(1, 10) 
train_errors = []
test_errors = []
X_train= df_mod.drop(f'{target_item}',axis=1).iloc[:1879]
X_test= df_mod.drop(f'{target_item}',axis=1).iloc[1879:]
y_train= df_mod[f'{target_item}'].iloc[:1879]
y_test = df_mod[f'{target_item}'].iloc[1879:]
tscv = TimeSeriesSplit(n_splits=5)
for depth in max_depths:
    model = RandomForestRegressor(max_depth=depth, n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict on train/test sets
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Store errors (Mean Absolute Error or RMSE)
    train_errors.append(mean_absolute_error(y_train, train_preds))
    test_errors.append(mean_absolute_error(y_test, test_preds))


# Plot results
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_errors, label="Train Error", marker='o')
plt.plot(max_depths, test_errors, label="Test Error", marker='o')
plt.xlabel("Max Depth")
plt.ylabel("Error (MAE)")
plt.legend()
plt.title("Train vs. Test Error Across Max Depths")
plt.show()

train_errors2 = []
test_errors2 = []
for depth in max_depths:
    train_errors_split = []
    test_errors_split = []

    model_cv = RandomForestRegressor(n_estimators=100, max_depth=depth, n_jobs=-1)
    for train_idx, test_idx in tscv.split(df_mod.drop(f'{target_item}',axis=1)):
        X_train_cv, X_test_cv = df_mod.drop(f'{target_item}',axis=1).iloc[train_idx], df_mod.drop(f'{target_item}',axis=1).iloc[test_idx]
        y_train_cv, y_test_cv = df_mod[f'{target_item}'].iloc[train_idx], df_mod[f'{target_item}'].iloc[test_idx]

        model_cv.fit(X_train_cv, y_train_cv) 

        train_preds = model_cv.predict(X_train_cv)
        test_preds = model_cv.predict(X_test_cv)

        train_errors_split.append(mean_absolute_error(y_train_cv, train_preds))
        test_errors_split.append(mean_absolute_error(y_test_cv, test_preds))
    
    train_errors2.append(np.mean(train_errors_split))
    test_errors2.append(np.mean(test_errors_split))

plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_errors2, label="Train Error", marker='o')
plt.plot(max_depths, test_errors2, label="Test Error", marker='o')
plt.xlabel("Max Depth")
plt.ylabel("Error (MAE)")
plt.legend()
plt.title("Train vs. Test Error Across Max Depths")
plt.show()