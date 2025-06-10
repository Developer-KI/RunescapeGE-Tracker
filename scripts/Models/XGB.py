#%%
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
#%%
from numpy import sqrt, mean, std, nan, concatenate, nanmean, nanstd
import optuna
import ModelTools as tools
import warnings
import scipy.stats as scistats
from statsmodels.tsa.stattools import adfuller

warnings.simplefilter("error")

def XGB(data:pd.DataFrame,target_col:str, holdout:int, recursive:bool=False, transform:str=None, n_estimators:int=200, time_splits:int=5, max_depth:int=5,learning_rate:int=0.03, subsample:int=0.4, colsample_bytree:int=0.8,min_child_weight=5):
    #*** recursive unimplented***
    full_X = data.drop(target_col, axis=1)
    full_Y = data[target_col]
    if transform != None:
        dtype = 'float64'
    else: dtype = 'float32'
    X = data.drop(target_col, axis=1).iloc[:-holdout].to_numpy(dtype=dtype)
    Y = data[target_col].iloc[:-holdout].to_numpy(dtype=dtype)

    tscv = TimeSeriesSplit(n_splits=time_splits)
    cv_mase=[]
    cv_mae=[]
    cv_da=[]
    best_mase=float('inf')
    best_mase=float('inf')
    best_model=None

    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X[train_idx,:], X[test_idx,:]
        y_train_cv, y_test_cv = Y[train_idx], Y[test_idx]

        if transform=='boxcox': #non-price spike pruning will cause integer overflow
            adf_count=0
            y_train_cv=scistats.boxcox(y_train_cv)
            adf_result =adfuller(y_train_cv)[1]
            if adf_result >0.05:
                adf_count=adf_count+1

        model_cv = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, min_child_weight=min_child_weight, colsample_bytree=colsample_bytree)
        model_cv.fit(X_train_cv, y_train_cv) 
        preds_cv = model_cv.predict(X_test_cv)

        model_mase = tools.mase(y_test_cv, preds_cv, y_train_cv,1)
        model_mae= mean_absolute_error(y_test_cv,preds_cv)
        cv_mase.append(model_mase)
        cv_mae.append(model_mae)

        if len(y_test_cv) > 0 and len(y_train_cv) > 0: # Ensure there's data to calculate direction
            last_actual_in_train_cv = y_train_cv[-1]
            
            # Combine for DA calculation
            combined_actuals_cv_da = concatenate(([last_actual_in_train_cv], y_test_cv))
            combined_preds_cv_da = concatenate(([last_actual_in_train_cv], preds_cv))
            
            model_da = tools.calculate_directional_accuracy(combined_actuals_cv_da, combined_preds_cv_da)
            cv_da.append(model_da)
        else:
            cv_da.append(nan) # Append NaN if not enough data for DA

        if model_mase < best_mase:
            best_mase = model_mase
            best_model = model_cv  
            
    if best_model is not None:
        print(f"Cross-validated MASE: {mean(cv_mase):.4f} (±{std(cv_mase):.4f})")
        print(f"Cross-validated MAE: {mean(cv_mae):.4f} (±{std(cv_mae):.4f})")
        print(f"Cross-validated DA: {nanmean(cv_da):.2f}% (±{nanstd(cv_da):.2f})") # Use nanmean/nanstd to handle NaNs
        if transform!= None:
            print(f"Aggregated Training Augmented Dickey-Fuller Rejection: {adf_count/time_splits}%")

        #**fix AI numpy conversion slop**
        # Final model training
        final_train_X = full_X.iloc[:-holdout].to_numpy(dtype='float32')
        final_train_Y = full_Y.iloc[:-holdout].to_numpy(dtype='float32')
        best_model.fit(final_train_X, final_train_Y)

        # Final model training
        final_train_X = full_X.iloc[:-holdout].to_numpy(dtype='float32')
        final_train_Y = full_Y.iloc[:-holdout].to_numpy(dtype='float32')
        best_model.fit(final_train_X, final_train_Y)
       
        if recursive==False:
            # Final holdout evaluation
            X_holdout = full_X.iloc[-holdout:].to_numpy(dtype='float32') 
            Y_holdout = full_Y.iloc[-holdout:].to_numpy(dtype='float32')
            final_preds_holdout = best_model.predict(X_holdout)

            final_holdout_mase = tools.mase(Y_holdout, final_preds_holdout, final_train_Y, 1)
            final_holdout_mae = mean_absolute_error(Y_holdout, final_preds_holdout)

            # Calculate Final Holdout DA
            # Get the last actual price from the training data before the holdout
            last_train_actual = full_Y.iloc[-holdout - 1] 
            
            # Combine last training actual with holdout actuals and predictions
            combined_actuals_for_da = concatenate(([last_train_actual], Y_holdout))
            combined_preds_for_da = concatenate(([last_train_actual], final_preds_holdout))
            
            final_holdout_da = tools.calculate_directional_accuracy(combined_actuals_for_da, combined_preds_for_da)
            print('--------------------------')
            print(f"Final Holdout MASE: {final_holdout_mase:.4f}")
            print(f"Final Holdout MAE: {final_holdout_mae:.4f}")
            print(f"Final Holdout Directional Accuracy: {final_holdout_da:.2f}%")

            return best_model, final_preds_holdout
        else: #WORK IN PROGRESS ------------------
            for i in range(0,len(holdout)+1):
                X_next = full_X.iloc[-holdout:-holdout+1].to_numpy(dtype='float32') 
                Y_next = full_Y.iloc[-holdout:-holdout+1].to_numpy(dtype='float32')
                final_preds_recursive = best_model.predict(X_next)
                X_next= concatenate(final_preds_recursive,X_next)
                Y_next= concatenate(final_preds_recursive,Y_next)
    else:
        raise Exception("Failed to choose model")
    
def XGBOptim(data:pd.DataFrame,target_col:str, holdout:int,n_trials:int=30,pruner:bool=True,meta_weight=False)-> dict:
    X = data.drop(target_col, axis=1).iloc[:-holdout].to_numpy(dtype="float32")
    Y = data[target_col].iloc[:-holdout].to_numpy(dtype="float32")

    def objective(trial):
        if meta_weight==True:
            alpha = trial.suggest_float("alpha", 0.1, 2.0)
        else: alpha=1
        tscv = TimeSeriesSplit(n_splits=5)
        mase_scores = []
        best_mase=float('inf')
        best_test_idx=None
        for train_idx, test_idx in tscv.split(X):
            
            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = Y[train_idx], Y[test_idx]

            model = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 20, 400), #50,200
                max_depth=trial.suggest_int("max_depth", 2, 50),
                eta=trial.suggest_int('eta',0.001,1),
                subsample=trial.suggest_int("subsample", 0, 1),
                colsample_bytree=trial.suggest_int("subsample", 0, 1)
                )

            model.fit(X_train_cv, y_train_cv)
            preds_cv = model.predict(X_test_cv)
            mase = tools.mase(y_test_cv, preds_cv,y_train_cv,1)
            mase_scores.append(mase)
            # trial.report(mean(mase_scores), step=len(mase_scores))  # Log progress
            # if pruner==True:
                # if trial.should_prune():
                #     raise optuna.TrialPruned()

            if mase < best_mase:
                best_mase = mase
                best_test_idx=test_idx
        trial.set_user_attr("best_test_idx", best_test_idx)
        trial.set_user_attr("mase_scores", mase_scores)
        return mean(mase_scores)*alpha+std(mase_scores)
    
    study = optuna.create_study(directions=["minimize"])
    study.optimize(objective, n_trials=n_trials, n_jobs=4) #check processor cores for parallelization 

    # Select first Pareto-optimal trial to train final model

    if not study.best_trials:
        raise Exception("Optuna failed to find optimal parameters.")

    best_params = study.best_trials[0].params.copy()
    best_alpha=best_params.pop("alpha", None) 
    best_test_idx= study.best_trials[0].user_attrs.get("best_test_idx")
    best_model = XGBRegressor(**best_params)
    best_model.fit(X,Y)

    print("------------------------Best Trial MASE and Standard Deviation------------------------")
    for trial in study.best_trials:
        print(f"MASE: {mean(trial.user_attrs['mase_scores'])} | Std Dev: {std(trial.user_attrs['mase_scores'])} | Ratio: {mean(trial.user_attrs['mase_scores'])/std(trial.user_attrs['mase_scores'])}")
        print(f'Weighted Score: {trial.values}\nParams: {trial.params}')
    print("-----------------------------------------------------------------------------------------")

    return best_model, best_params, best_test_idx
