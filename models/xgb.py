#%%
import pandas as pd
import numpy as np
import optuna
from   warnings import simplefilter
from   xgboost import XGBRegressor
from   sklearn.model_selection import TimeSeriesSplit
from   sklearn.metrics import mean_absolute_error
from   utils import model_tools as tools
import utils.outlier_detection as outlier
#import scipy.stats as scistats
#from statsmodels.tsa.stattools import adfuller

simplefilter("error")
#%%
def train_xgb_model(
    data:               pd.DataFrame,
    target_col:         str,
    holdout:            int,
    time_splits:        int = 5,
    n_estimators:       int = 100,
    max_depth:          int = 10,
    learning_rate:      float=0.03, 
    subsample:          float=0.4, 
    colsample_bytree:   float=0.8, 
    min_child_weight:   int=5
) -> tuple: #implement recursive predictions
    dtype = 'float32'
    
    full_x, full_y = tools.prep_tree_model(data, target_col, holdout)
    train_x = data.drop(target_col, axis=1).iloc[:-holdout].to_numpy(dtype=dtype)
    train_y = data[target_col].iloc[:-holdout].to_numpy(dtype=dtype)

    tscv = TimeSeriesSplit(n_splits=time_splits)
    cv_mase=[]
    cv_mae=[]
    cv_da=[]
    best_mase=float('inf')
    best_mase=float('inf')
    best_model=None

    for train_idx, test_idx in tscv.split(train_x):
        X_train_cv, X_test_cv = train_x[train_idx,:], train_x[test_idx,:]
        y_train_cv, y_test_cv = train_y[train_idx], train_y[test_idx]

        #----- buggy mess
        # if transform=='boxcox': #non-price spike pruning will cause integer overflows
        #     y_train_cv=y_train_cv.astype(np.float64)
        #     adf_count=0
        #     y_train_cv, _ = boxcox(y_train_cv) #trashed tuple lambda value return
        #     print(y_train_cv)
        #     adf_result =adfuller(y_train_cv)[1]
        #     if adf_result >0.05:
        #         adf_count=adf_count+1

        model_cv = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, min_child_weight=min_child_weight, colsample_bytree=colsample_bytree, n_jobs=-1)
        model_cv.fit(X_train_cv, y_train_cv) 
        preds_cv = model_cv.predict(X_test_cv)

        model_mase = tools.mase(y_test_cv, preds_cv, y_train_cv,1)
        model_mae= mean_absolute_error(y_test_cv,preds_cv)
        cv_mase.append(model_mase)
        cv_mae.append(model_mae)

        if len(y_test_cv) > 0 and len(y_train_cv) > 0: # Ensure there's data to calculate direction
            last_actual_in_train_cv = y_train_cv[-1]
            
            # Combine for DA calculation
            combined_actuals_cv_da = np.concatenate(([last_actual_in_train_cv], y_test_cv))
            combined_preds_cv_da = np.concatenate(([last_actual_in_train_cv], preds_cv))
            
            model_da = tools.calculate_directional_accuracy(combined_actuals_cv_da, combined_preds_cv_da)
            cv_da.append(model_da)
        else:
            cv_da.append(np.nan) # Append NaN if not enough data for DA

        if model_mase < best_mase:
            best_mase = model_mase
            best_model = model_cv  
            
    if best_model is not None:
        
        # if transform!= None:
        #     print(f"Aggregated Training Augmented Dickey-Fuller Rejection: {adf_count/time_splits}%")

        #**fix AI numpy conversion slop**
        # Final model training
        final_train_X = full_x.iloc[:-holdout].to_numpy(dtype='float32')
        final_train_Y = full_y.iloc[:-holdout].to_numpy(dtype='float32')
        best_model.fit(final_train_X, final_train_Y)

        # Final holdout evaluation
        x_holdout = full_x.iloc[-holdout:].to_numpy(dtype='float32') 
        
        final_preds_holdout = best_model.predict(x_holdout)
        
        return best_model, final_preds_holdout, final_train_Y, cv_mae, cv_mase, cv_da
    else:
        raise Exception("Failed to choose model.")

def XGB(
    data:               pd.DataFrame,
    target_col:         str, 
    holdout:            int, 
    outlier_threshold:  float = 2,
    outlier_window:     int = 20,
    detection:          str = 'ewm',
    ewm_bounds:         list[float]|None = None,
    n_estimators:       int = 200, 
    time_splits:        int = 5, 
    max_depth:          int = 5, 
    learning_rate:      float = 0.03, 
    subsample:          float = 0.4, 
    colsample_bytree:   float = 0.8, 
    min_child_weight:   int = 5
) -> tuple:
    
    full_y = data[target_col]

    if detection=='rolling-z':
        outliers = outlier.rolling_zscore(full_y, outlier_window, outlier_threshold)
        print(f'Total Outliers Detected: {len(outliers)}\n--------------------------')
    elif detection=='iqr':
        outliers = outlier.iqr(full_y, outlier_threshold)
        print(f'Total Outliers Detected: {len(outliers)}\n--------------------------')
    elif detection=='ewm' and ewm_bounds is not None:
        ewm_bounds_percentage_lower = 1-ewm_bounds[0]
        ewm_bounds_percentage_upper = 1+ewm_bounds[1]
        outliers = outlier.ewm(full_y, outlier_window,ewm_bounds_percentage_lower,ewm_bounds_percentage_upper)
        print(f'Total Outliers Detected: {len(outliers)}\n--------------------------')
    elif detection is not None and not isinstance(detection, str):
        raise ValueError("Outlier parameter must be of string type")
    else: raise ValueError("Selection error")
    
    full_y_filtered = full_y.drop(outliers.index)
    y_holdout = full_y_filtered.iloc[-holdout:].to_numpy(dtype='float32')

    best_model, final_preds_holdout, final_train_Y, cv_mae, cv_mase, cv_da = train_xgb_model(
        data, 
        target_col, 
        holdout, 
        time_splits, 
        n_estimators, 
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        min_child_weight
        )
    final_holdout_mae, final_holdout_mase, final_holdout_da = tools.score_tree_model(full_y, holdout, y_holdout, final_preds_holdout, final_train_Y)

    print(f"Cross-validated MASE: {np.mean(cv_mase):.4f} (±{np.std(cv_mase):.4f})")
    print(f"Cross-validated MAE: {np.mean(cv_mae):.4f} (±{np.std(cv_mae):.4f})")
    print(f"Cross-validated DA: {np.nanmean(cv_da):.2f}% (±{np.nanstd(cv_da):.2f})") # Use nanmean/nanstd to handle NaNs

    #*** recursive unimplented***

    print('--------------------------')
    print(f"Final Holdout MASE: {final_holdout_mase:.4f}")
    print(f"Final Holdout MAE: {final_holdout_mae:.4f}")
    print(f"Final Holdout Directional Accuracy: {final_holdout_da:.2f}%")
    if detection is not None and isinstance(detection,str):
        return best_model, final_preds_holdout, outliers
    else: return best_model, final_preds_holdout

def XGBOptim(
    data:pd.DataFrame,
    target_col:str, 
    holdout:int, 
    n_trials:int=30,
    pruner:bool=True,
    meta_weight=False
)-> tuple:
    
    X = data.drop(target_col, axis=1).iloc[:-holdout].to_numpy(dtype="float32")
    Y = data[target_col].iloc[:-holdout].to_numpy(dtype="float32")

    def objective(trial) -> float:
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
        return float(np.mean(mase_scores)*alpha+np.std(mase_scores))
    
    study = optuna.create_study(direction="minimize")
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
        print(f"MASE: {np.mean(trial.user_attrs['mase_scores'])} | Std Dev: {np.std(trial.user_attrs['mase_scores'])} | Ratio: {np.mean(trial.user_attrs['mase_scores'])/np.std(trial.user_attrs['mase_scores'])}")
        print(f'Weighted Score: {trial.values}\nParams: {trial.params}')
    print("-----------------------------------------------------------------------------------------")

    return best_model, best_params, best_test_idx
