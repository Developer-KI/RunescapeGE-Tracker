#%%
import numpy as np
import pandas as pd
from   hmmlearn.hmm import MultinomialHMM
from   utils.model_tools import plot_classification_vs_price, rolling_threshold_classification
from   sklearn.preprocessing import OneHotEncoder
#%%
def hmm_data_prep(raw_features:pd.DataFrame, item:str, window:int=100, diff_percent:float=0.1) -> tuple: 
    n_features= raw_features.shape[1]
    n_samples= raw_features.shape[0]
    n_components= raw_features.shape[1]

    boolean_features = rolling_threshold_classification(raw_features,window,diff_percent)

    features=boolean_features[item].values.reshape(-1,1)
    features[0,0]=2
    n_components=len(np.unique(features))
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    features_encoded = encoder.fit_transform(features).astype(int)[:,0:2]  # Shape will now be (2499, 3)

    return n_components, n_features, n_samples, features_encoded

def hmm_train_score (features_encoded:pd.DataFrame, n_components:int, n_features:int, iter:int) -> tuple:
    hmm_model = MultinomialHMM(n_components=n_components, n_iter=iter) #unimplemented: leave init_params empty to self-select probabilities
    hmm_model.fit(features_encoded)
    hidden_states= hmm_model.predict(features_encoded)
    log_likelihood = hmm_model.score(features_encoded)
    n_parameters = n_components**2 +(n_components*n_features) + n_components # Estimate number of parameters
    
    return hidden_states, log_likelihood, n_parameters, hmm_model

def item_threshold_hmm(features: pd.DataFrame, item: str, iter:int=1000, window:int=100, diff_percent:float=0.1) -> tuple:
#self-select not implemented
#Paramters for price differences governing regime change
    n_components, n_features, n_samples, features_encoded = hmm_data_prep(features, item, window, diff_percent)
    hidden_states, log_likelihood, n_parameters, hmm_model = hmm_train_score(features_encoded, n_components, n_features, iter)

    aic = 2 * n_parameters - 2 * log_likelihood
    bic = n_parameters * np.log(n_samples) - 2 * log_likelihood
    
    print(f"AIC: {aic}, BIC: {bic}")
    plot_classification_vs_price(features,hidden_states,item,hmm_model)

    return [aic,bic], hmm_model

#%%


    # #startprob = np.array([.17,.66,.17]) #reasonable to keep up/down always less than sideways
    # #transprob = np.array([[.17,.66,.17],[.17,.66,.17],[.17,.66,.17]])
    # #emissionprob = np.array([[.17,.66,.17],[.17,.66,.17],[.17,.66,.17]])
    # #hmm_model = MultinomialHMM(n_components=n_components, startprob_prior=startprob, transmat_prior=transprob, n_iter=iter) #leave init_params empty to self-select probabilities
    # #hmm_model.emissionprob_ = np.array(emissionprob)