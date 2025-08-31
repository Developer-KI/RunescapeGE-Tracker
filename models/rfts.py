import pandas as pd
import numpy as np
import optuna
from   sklearn.ensemble import RandomForestRegressor
from   sklearn.model_selection import TimeSeriesSplit
from   sklearn.metrics import mean_absolute_error
from   utils import model_tools as tools
import utils.outlier_detection as outlier
from   statsmodels.tsa.stattools import adfuller
from   typing import Literal
#from   scipy.stats import boxcox
from   warnings import simplefilter

simplefilter("error")

def _train_rfts_model(
    train_x:            pd.Series,
    train_y:            pd.Series,
    time_splits:        int = 5,
    outlier_threshold:  float|None = 2,
    outlier_window:     int = 20,
    detection:          str = 'ewm',
    ewm_bounds:         list[float]|None = None,
    mase_m:             int = 1,
    min_samples_leaf:   int = 5,
    min_samples_split:  int = 5,
    n_estimators:       int = 100,
    max_features:       float | Literal['sqrt', 'log2'] = 'sqrt',
    max_depth:          int = 10,
) -> tuple:
    
    tscv = TimeSeriesSplit(n_splits=time_splits)
    
    train_cv_mase=[]
    train_cv_mae=[]
    cv_mase=[]
    cv_mae=[]
    cv_da=[]
    
    outliers_set=set()
    best_mase=float('inf')
    best_mase=float('inf')
    best_model=None

    for train_idx, test_idx in tscv.split(train_x):
        x_train_cv, x_test_cv = train_x.iloc[train_idx], train_x.iloc[test_idx]
        y_train_cv, y_test_cv = train_y.iloc[train_idx], train_y.iloc[test_idx]

        if detection=='rolling-z':
            y_train_outliers = outlier.rolling_zscore(y_train_cv, outlier_window, outlier_threshold)
            y_test_outliers = outlier.rolling_zscore(y_test_cv, outlier_window, outlier_threshold)
        elif detection=='iqr':
            y_train_outliers = outlier.iqr(y_train_cv, outlier_threshold)
            y_test_outliers = outlier.iqr(y_test_cv, outlier_threshold)
        elif detection=='ewm2' and ewm_bounds is not None:
            ewm_bounds_lower = ewm_bounds[0]
            ewm_bounds_upper = ewm_bounds[1]
            y_train_outliers = outlier.ewm_z_residuals2(y_train_cv, outlier_window,ewm_bounds_lower,ewm_bounds_upper, True)
            y_test_outliers = outlier.ewm_z_residuals2(y_test_cv, outlier_window,ewm_bounds_lower,ewm_bounds_upper, True)
        elif detection=='ewm' and ewm_bounds is not None:
            ewm_bounds_lower = ewm_bounds[0]
            ewm_bounds_upper = ewm_bounds[1]
            y_train_outliers = outlier.ewm_z_residuals(y_train_cv, outlier_window,ewm_bounds_lower,ewm_bounds_upper)
            y_test_outliers = outlier.ewm_z_residuals(y_test_cv, outlier_window,ewm_bounds_lower,ewm_bounds_upper)
        elif detection is not None and not isinstance(detection, str):
            raise ValueError("Outlier parameter must be of string type")
        elif detection is None:
            pass
        else: raise ValueError("Selection error")
            
        
        if detection:
            outliers_set.update(y_train_outliers.index)
            outliers_set.update(y_test_outliers.index)

            # Filter both X and y based on the outliers found in y
            x_train_cv_filtered = x_train_cv[~np.isin(np.arange(len(y_train_cv)), y_train_outliers.index)]
            y_train_cv_filtered = y_train_cv[~np.isin(np.arange(len(y_train_cv)), y_train_outliers.index)]
            x_test_cv_filtered = x_test_cv[~np.isin(np.arange(len(y_test_cv)), y_test_outliers.index)]
            y_test_cv_filtered = y_test_cv[~np.isin(np.arange(len(y_test_cv)), y_test_outliers.index)]
        else:
            x_train_cv_filtered = x_train_cv
            y_train_cv_filtered = y_train_cv
            x_test_cv_filtered = x_test_cv
            y_test_cv_filtered = y_test_cv


        model_cv = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            max_features=max_features, 
            n_jobs=-1
            )
        model_cv.fit(x_train_cv_filtered, y_train_cv_filtered) 

        preds_cv_filtered = model_cv.predict(x_test_cv_filtered)
        model_mase = tools.mase(y_test_cv_filtered, y_train_cv_filtered, preds_cv_filtered, mase_m)
        model_mae= mean_absolute_error(y_test_cv_filtered,preds_cv_filtered)
        cv_mase.append(model_mase)
        cv_mae.append(model_mae)
        
        preds_train_cv_filtered = model_cv.predict(x_train_cv_filtered)
        model_train_mase = tools.mase(y_train_cv_filtered, y_train_cv_filtered, preds_train_cv_filtered,mase_m)
        model_train_mae= mean_absolute_error(y_train_cv_filtered,preds_train_cv_filtered)
        train_cv_mase.append(model_train_mase)
        train_cv_mae.append(model_train_mae)

        if len(y_test_cv_filtered) > 0 and len(y_train_cv_filtered) > 0: # Ensure there's data to calculate direction
            last_actual_in_train_cv = y_train_cv_filtered.iloc[-1]
            
            # Combine for DA calculation
            combined_actuals_cv_da = np.concatenate(([last_actual_in_train_cv], y_test_cv_filtered))
            combined_preds_cv_da = np.concatenate(([last_actual_in_train_cv], preds_cv_filtered))
            
            model_da = tools.calculate_directional_accuracy(combined_actuals_cv_da, combined_preds_cv_da)
            cv_da.append(model_da)
        else:
            cv_da.append(np.nan) # Append NaN if not enough data for DA

        if model_mase < best_mase:
            best_mase = model_mase
            best_model = model_cv  
            
    if best_model is not None:
        final_outliers = sorted(list(outliers_set))
        return best_model, final_outliers, cv_mae, cv_mase, cv_da, train_cv_mae, train_cv_mase
    else:
        raise Exception("Failed to choose model.")

def RFTS(
    data:               pd.DataFrame,
    target_col:         str,
    holdout:            int,
    outlier_threshold:  float|None = 2,
    outlier_window:     int = 20,
    detection:          str = 'ewm',
    ewm_bounds:         list[float]|None = None,
    mase_m:             int = 1,
    time_splits:        int = 5,
    min_samples_leaf:   int = 5,
    min_samples_split:  int = 5,
    n_estimators:       int = 100,
    max_features:       float | Literal['sqrt', 'log2'] = 'sqrt',
    max_depth:          int = 10
) -> tuple:

    train_x, train_y, holdout_x, holdout_y = tools.prep_tree_model(data, target_col, holdout)

    best_model, train_outliers_idx, cv_mae, cv_mase, cv_da, train_cv_mae, train_cv_mase  = _train_rfts_model(
        train_x,
        train_y,  
        time_splits, 
        outlier_threshold,  
        outlier_window,  
        detection,          
        ewm_bounds, 
        mase_m,        
        min_samples_leaf, 
        min_samples_split, 
        n_estimators, 
        max_features, 
        max_depth
        )

    if detection=='rolling-z':
        holdout_outliers_idx = outlier.rolling_zscore(holdout_y, outlier_window, outlier_threshold).index
        print(f'Holdout Outliers Detected: {len(holdout_outliers_idx)}\n--------------------------')
    elif detection=='iqr':
        holdout_outliers_idx = outlier.iqr(holdout_y, outlier_threshold).index
        print(f'Holdout Outliers Detected: {len(holdout_outliers_idx)}\n--------------------------')
    elif detection=='ewm2' and ewm_bounds is not None:
        ewm_bounds_lower = ewm_bounds[0]
        ewm_bounds_upper = ewm_bounds[1]
        holdout_outliers_idx = outlier.ewm_z_residuals2(holdout_y, outlier_window,ewm_bounds_lower,ewm_bounds_upper, True).index
        print(f'Holdout Outliers Detected: {len(holdout_outliers_idx)}\n--------------------------')
    elif detection=='ewm' and ewm_bounds is not None:
        ewm_bounds_lower = ewm_bounds[0]
        ewm_bounds_upper = ewm_bounds[1]
        holdout_outliers_idx = outlier.ewm_z_residuals(holdout_y, outlier_window,ewm_bounds_lower,ewm_bounds_upper).index
        print(f'Holdout Outliers Detected: {len(holdout_outliers_idx)}\n--------------------------')
    elif detection is not None and not isinstance(detection, str):
        raise ValueError("Outlier parameter must be of string type")
    elif detection is None:
        pass
    else: raise ValueError("Selection error")

    if detection: 
        holdout_outliers_idx = list(holdout_outliers_idx)
        train_y_filtered = train_y.drop(train_outliers_idx)
        train_x_filtered = train_x.drop(train_outliers_idx)
        holdout_x_filtered = holdout_x.drop(holdout_outliers_idx)
        holdout_y_filtered = holdout_y.drop(holdout_outliers_idx)
    else:
        train_y_filtered = train_y
        train_x_filtered = train_x
        holdout_x_filtered = holdout_x
        holdout_y_filtered = holdout_y
    best_model.fit(train_x_filtered, train_y_filtered)
    holdout_preds = best_model.predict(holdout_x_filtered)

    final_holdout_mae, final_holdout_mase, final_holdout_da = tools.score_tree_model(train_y_filtered, holdout_y_filtered, holdout_preds)
    print(f"Cross-validated MASE: {np.mean(cv_mase):.4f} (±{np.std(cv_mase):.4f})")
    print(f"Cross-validated MAE: {np.mean(cv_mae):.4f} (±{np.std(cv_mae):.4f})")
    print(f"Cross-validated DA: {np.nanmean(cv_da):.2f}% (±{np.nanstd(cv_da):.2f})") # Use nanmean/nanstd to handle NaNs

    #*** recursive unimplented***

    print('--------------------------')
    print(f"Final Holdout MASE: {final_holdout_mase:.4f}")
    print(f"Final Holdout MAE: {final_holdout_mae:.4f}")
    print(f"Final Holdout Directional Accuracy: {final_holdout_da:.2f}%")
    if detection is not None and isinstance(detection,str):
        return best_model, holdout_preds, train_outliers_idx, holdout_outliers_idx, cv_mae, cv_mase, train_cv_mae, train_cv_mase
    else: return best_model, holdout_preds, None, None, cv_mae, cv_mase, train_cv_mae, train_cv_mase
        
def RFTSOptim(data:pd.DataFrame,target_col:str, holdout:int,n_trials:int=30,pruner:bool=True,meta_weight=False)-> tuple:
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

            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 20, 400), #50,200
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                max_depth=trial.suggest_int('max_depth', 1,7),
                n_jobs=-1
                )

            model.fit(X_train_cv, y_train_cv)
            preds_cv = model.predict(X_test_cv)
            mase = tools.mase(y_test_cv, y_train_cv, preds_cv, 1)
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
        # return float(np.mean(mase_scores)*alpha+np.std(mase_scores))
        return float(np.mean(mase_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=4) #check processor cores for parallelization 

    # Select first Pareto-optimal trial to train final model

    if not study.best_trials:
        raise Exception("Optuna failed to find optimal parameters.")

    best_params = study.best_trials[0].params.copy()
    best_alpha=best_params.pop("alpha", None) 
    best_test_idx= study.best_trials[0].user_attrs.get("best_test_idx")
    best_model = RandomForestRegressor(**best_params, n_jobs=-1)
    best_model.fit(X,Y)

    print("------------------------Best Trial MASE and Standard Deviation------------------------")
    for trial in study.best_trials:
        print(f"MASE: {np.mean(trial.user_attrs['mase_scores'])} | Std Dev: {np.std(trial.user_attrs['mase_scores'])} | Ratio: {np.mean(trial.user_attrs['mase_scores'])/np.std(trial.user_attrs['mase_scores'])}")
        print(f'Weighted Score: {trial.values}\nParams: {trial.params}')
    print("-----------------------------------------------------------------------------------------")

    return best_model, best_params, best_test_idx





