#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from hmmlearn.hmm import MultinomialHMM
import ModelTools as tools
from sklearn.preprocessing import OneHotEncoder

def ItemThresholdHMM(features,item,iter=100,window=100,diffpercent=0.1, selfselect:pd.array= None):
#self-select not implemented
#Paramters for price differences governing regime change
#Window must be >0, 1= no window

    n_features= features.shape[1]
    n_samples= features.shape[0]
    n_components= features.shape[1]

    booleandf = tools.rolling_threshold_classification(features,window,diffpercent)

    X=booleandf[item].values.reshape(-1,1)
    X[0,0]=2
    n_components=len(np.unique(X))
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    X_encoded = encoder.fit_transform(X).astype(int)  # Shape will now be (2499, 3)

    if selfselect==None:
        HMMmodel = MultinomialHMM(n_components=n_components, n_iter=iter) #leave init_params empty to self-select probabilities
        HMMmodel.fit(X_encoded)
        hidden_states= HMMmodel.predict(X_encoded)

        log_likelihood = HMMmodel.score(X_encoded)
        # Estimate number of parameters
        num_parameters = n_components**2 +(n_components*n_features) + n_components
        # Compute AIC & BIC
        aic = 2 * num_parameters - 2 * log_likelihood
        bic = num_parameters * np.log(n_samples) - 2 * log_likelihood
        print(f"AIC: {aic}, BIC: {bic}")

        tools.plot_classification_vs_price(features,X_encoded,item,HMMmodel)
        return [aic,bic], HMMmodel

#%%


    # n_features= features.shape[1]
    # n_samples= features.shape[0]
    # #startprob = np.array([.17,.66,.17]) #reasonable to keep up/down always less than sideways
    # #transprob = np.array([[.17,.66,.17],[.17,.66,.17],[.17,.66,.17]])
    # #emissionprob = np.array([[.17,.66,.17],[.17,.66,.17],[.17,.66,.17]])
    # #HMMmodel = MultinomialHMM(n_components=n_components, startprob_prior=startprob, transmat_prior=transprob, n_iter=iter) #leave init_params empty to self-select probabilities
    # #HMMmodel.emissionprob_ = np.array(emissionprob)

    # if selfselect==None:
    #     HMMmodel = MultinomialHMM(n_components=n_components, n_iter=iter) #leave init_params empty to self-select probabilities
    #     HMMmodel.fit(features)
    #     hidden_states= HMMmodel.predict(features)

    #     log_likelihood = HMMmodel.score(features)
    #     # Estimate number of parameters
    #     num_parameters = n_components**2 +(n_components*n_features) + n_components
    #     # Compute AIC & BIC
    #     aic = 2 * num_parameters - 2 * log_likelihood
    #     bic = num_parameters * np.log(n_samples) - 2 * log_likelihood

    #     print(f"AIC: {aic}, BIC: {bic}")
    #     return [aic,bic], HMMmodel


    # booleandf = tools.rolling_threshold_classification(price_matrix_items,50,0.1)
    # item=207


    # X=booleandf[item].values.reshape(-1,1)
    # X[0,0]=2
    # n_components=len(np.unique(X))
    # encoder = OneHotEncoder(sparse_output=False, categories='auto')
    # X_encoded = encoder.fit_transform(X).astype(int)  # Shape will now be (2499, 3)

    # iter = 198
    # #startprob = np.array([.17,.66,.17]) #reasonable to keep up/down always less than sideways
    # #transprob = np.array([[.17,.66,.17],[.17,.66,.17],[.17,.66,.17]])
    # #emissionprob = np.array([[.17,.66,.17],[.17,.66,.17],[.17,.66,.17]])
    # #HMMmodel = MultinomialHMM(n_components=n_components, startprob_prior=startprob, transmat_prior=transprob, n_iter=iter) #leave init_params empty to self-select probabilities
    # HMMmodel = MultinomialHMM(n_components=n_components, n_iter=iter) #leave init_params empty to self-select probabilities
    # #HMMmodel.emissionprob_ = np.array(emissionprob)
    # HMMmodel.fit(X_encoded)
    # hidden_states= HMMmodel.predict(X_encoded)

    # log_likelihood = HMMmodel.score(X_encoded)
    # # Estimate number of parameters
    # num_parameters = n_components**2 +(n_components*X_encoded.shape[1]) + n_components
    # # Compute AIC & BIC
    # aic = 2 * num_parameters - 2 * log_likelihood
    # bic = num_parameters * np.log(X.shape[0]) - 2 * log_likelihood

    # print(f"AIC: {aic}, BIC: {bic}")

    # tools.plot_classification_vs_price(price_matrix_items,X_encoded,item,HMMmodel)