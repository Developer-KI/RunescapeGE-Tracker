import pandas as pd
from numpy import sqrt, mean, std 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

def RFTS(data: pd.DataFrame, target_col: str, splits: int = 5, estimators: int = 100, seed: int = 42) -> RandomForestRegressor:
    X = data.drop(target_col, axis=1)
    Y = data[target_col]

    tscv = TimeSeriesSplit(n_splits=splits)
    cv_mse = []
    cv_mae = []
    output_model = None

    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = Y.iloc[train_idx], Y.iloc[test_idx]
    
        model_cv = RandomForestRegressor(n_estimators=estimators, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=seed, n_jobs=-1)
        model_cv.fit(X_train_cv, y_train_cv)
    
        preds_cv = model_cv.predict(X_test_cv)

        model_mse = mean_squared_error(y_test_cv, preds_cv)
        model_mae = mean_absolute_error(y_test_cv, preds_cv)
        
        
        if model_mse < min(cv_mse, default=float('inf')):
            output_model = (model_cv, test_idx)

        cv_mse.append(model_mse)
        cv_mae.append(model_mae)
    
    if output_model != None:
        print(f"Cross-validated RMSE: {sqrt(mean(cv_mse)):.4f} (±{sqrt(std(cv_mse)):.4f})")
        print(f"Cross-validated MAE: {sqrt(mean(cv_mae)):.4f} (±{sqrt(std(cv_mae)):.4f})")
        return output_model
    else:
        raise Exception("Failed to choose model")