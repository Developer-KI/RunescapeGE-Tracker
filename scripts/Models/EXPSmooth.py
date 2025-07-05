import numpy as np
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing

def EXPSmooth(data, holdout:int, predictions:int=None, initialization_method:str=None):
    data.index=pd.to_datetime(data.index,unit='s')
    data = data.asfreq('5min')
    train=data.iloc[:holdout,0]
    model = SimpleExpSmoothing(train, initialization_method=initialization_method)
    fitted=model.fit()
    if predictions is None:
        predictions=holdout
    forecast=fitted.forecast(predictions)
    return model, forecast