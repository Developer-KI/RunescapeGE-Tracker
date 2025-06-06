import pandas as pd
from numpy import sqrt, mean, std 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import ModelTools as tools
import warnings
warnings.simplefilter("error")

def RFTS(data:pd.DataFrame,target_col:str,time_splits:int=5,min_samples_leaf:int=5,n_estimators:int=100,min_samples_split:int=5,max_features:str='sqrt',max_depth:int=10) -> RandomForestRegressor:
    X = data.drop(target_col, axis=1).to_numpy(dtype='float32')
    Y = data[target_col].to_numpy(dtype='float32')

    tscv = TimeSeriesSplit(n_splits=time_splits)
    cv_mase=[]
    best_mase=float('inf')
    best_model=None
    best_test_idx=None

    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X[train_idx,:], X[test_idx,:]
        y_train_cv, y_test_cv = Y[train_idx], Y[test_idx]

        model_cv = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)
        model_cv.fit(X_train_cv, y_train_cv) 
        preds_cv = model_cv.predict(X_test_cv)

        model_mase = tools.mase(y_test_cv, preds_cv, y_train_cv,1)
        cv_mase.append(model_mase)
        
        if model_mase < best_mase:
            best_mase = model_mase
            best_model = (model_cv, test_idx)  
            best_test_idx = test_idx
    
    if best_model != None:
        print(f"Cross-validated MASE: {mean(cv_mase):.4f} (±{std(cv_mase):.4f}) Ratio: {(mean(cv_mase)/std(cv_mase)):.4f}")
        return best_model
    else:
        raise Exception("Failed to choose model")
    
def RFTSOptim(data:pd.DataFrame,target_col:str,n_trials:int=30,pruner:bool=True,weight_ratio:int=1)-> dict:
    X = data.drop(target_col, axis=1).to_numpy(dtype="float32")
    Y = data[target_col].to_numpy(dtype="float32")

    def objective(trial):
        tscv = TimeSeriesSplit(n_splits=trial.suggest_int('n_TSsplits',2,50))
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
        return (mean(mase_scores)*weight_ratio+std(mase_scores)) # Optimize on average MSE across splits
        # or try weighted single combination mean_mse + x*var_mse
    
    study = optuna.create_study(directions=["minimize"])
    study.optimize(objective, n_trials=n_trials)

    # Select first Pareto-optimal trial to train final model

    if not study.best_trials:
        raise Exception("Optuna failed to find optimal parameters.")

    best_params = study.best_trials[0].params.copy()
    best_n_TSsplits=best_params.pop('n_TSsplits')
    best_test_idx= study.best_trials[0].user_attrs.get("best_test_idx")
    best_model = RandomForestRegressor(**best_params, n_jobs=-1)
    best_model.fit(X,Y)

    print("------------------------Best Trial MASE and Standard Deviation------------------------")
    for trial in study.best_trials:
        print(f"MASE: {mean(trial.user_attrs['mase_scores'])} | Std Dev: {std(trial.user_attrs['mase_scores'])} | Ratio: {mean(trial.user_attrs['mase_scores'])/std(trial.user_attrs['mase_scores'])}")
        print(f'Weighted Score: {trial.values}\nParams: {trial.params}')
    print("-----------------------------------------------------------------------------------------")

    return best_model, best_params, best_test_idx





