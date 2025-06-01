import pandas as pd
from numpy import sqrt, mean, std 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

def RFTS(data:pd.DataFrame,target_col:str,time_splits:int=5,min_samples_leaf:int=1,n_estimators:int=100,min_samples_split:int=2,max_features:str='sqrt',max_depth:int=None) -> RandomForestRegressor:
    X = data.drop(target_col, axis=1)
    Y = data[target_col]

    tscv = TimeSeriesSplit(n_splits=time_splits)
    cv_mse=[]
    cv_mae=[]
    best_mse=float('inf')
    best_model=None
    best_test_idx=None

    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = Y.iloc[train_idx], Y.iloc[test_idx]
    
        model_cv = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)
        model_cv.fit(X_train_cv, y_train_cv)
    
        preds_cv = model_cv.predict(X_test_cv)

        model_mse = mean_squared_error(y_test_cv, preds_cv)
        model_mae = mean_absolute_error(y_test_cv, preds_cv)
        cv_mse.append(model_mse)
        cv_mae.append(model_mae)
        
        
        if model_mse < best_mse:
            best_mse = model_mse
            best_model = (model_cv, test_idx)  
            best_test_idx = test_idx
    
    if best_model != None:
        print(f"Cross-validated RMSE: {sqrt(mean(cv_mse)):.4f} (±{sqrt(std(cv_mse)):.4f})")
        print(f"Cross-validated MAE: {mean(cv_mae):.4f} (±{std(cv_mae):.4f})")
        return best_model
    else:
        raise Exception("Failed to choose model")
    
def RFTSOptim(data:pd.DataFrame,target_col:str,time_splits:int=5,n_trials:int=30)-> dict:
    X = data.drop(target_col, axis=1)
    Y = data[target_col]
    tscv = TimeSeriesSplit(n_splits=time_splits)
    def objective(trial):
        mse_scores = []
        for train_idx, test_idx in tscv.split(X):
            
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cv, y_test_cv = Y.iloc[train_idx], Y.iloc[test_idx]

            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 20, 400), #50,200
                min_samples_split=trial.suggest_int("min_samples_split", 2, 15),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                max_depth=trial.suggest_int('max_depth', 1,50),
                n_jobs=-1
                )

            model.fit(X_train_cv, y_train_cv)
            preds_cv = model.predict(X_test_cv)
            mse_scores.append(mean_squared_error(y_test_cv, preds_cv))

        return mean(mse_scores), std(mse_scores)  # Optimize on average MSE across splits
        # or try weighted single combination mean_mse + x*var_mse
    
    study = optuna.create_study(directions=["minimize",'minimize'], sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)

    # Select first Pareto-optimal trial to train final model
    best_params = study.best_trials[0].params
    best_model = RandomForestRegressor(**best_params, n_jobs=-1)
    best_model.fit(X, Y)

    print("------------------------Best Trials RMSE and Standard Deviation------------------------")
    for trial in study.best_trials:
        print(f"RMSE: {trial.values[0]:.4f}, Std Dev: {trial.values[1]:.4f}\nParams: {trial.params}")
    print("-----------------------------------------------------------------------------------------")

    return {"Best Model": best_model, "Best Params": best_params}




