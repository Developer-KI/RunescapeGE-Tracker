#%%
import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from src.utils.plot_tools import plot_classification_vs_price
from src.utils.model_tools import rolling_threshold_classification
#%%
def hmm_data_prep(raw_features: pd.DataFrame, item: str, window: int = 100, diff_percent: float = 0.1) -> tuple:
    n_samples = raw_features.shape[0]

    classified = rolling_threshold_classification(raw_features, window, diff_percent)
    observations = classified[item].values.reshape(-1, 1)  # CategoricalHMM expects (n_samples, 1)
    n_symbols = len(np.unique(observations))

    return n_symbols, n_samples, observations

def hmm_train_score(observations: np.ndarray, n_components: int, n_symbols: int, n_iter: int) -> tuple:
    hmm_model = CategoricalHMM(n_components=n_components, n_iter=n_iter)
    hmm_model.fit(observations)
    hidden_states = hmm_model.predict(observations)
    log_likelihood = hmm_model.score(observations)
    # transition matrix + emission matrix + start probabilities
    n_parameters = n_components ** 2 + (n_components * n_symbols) + n_components

    return hidden_states, log_likelihood, n_parameters, hmm_model

def item_threshold_hmm(features: pd.DataFrame, item: str, n_components: int = 3,
                       n_iter: int = 1000, window: int = 100, diff_percent: float = 0.1) -> tuple:
    n_symbols, n_samples, observations = hmm_data_prep(features, item, window, diff_percent)
    hidden_states, log_likelihood, n_parameters, hmm_model = hmm_train_score(
        observations, n_components, n_symbols, n_iter
    )

    aic = 2 * n_parameters - 2 * log_likelihood
    bic = n_parameters * np.log(n_samples) - 2 * log_likelihood

    print(f"AIC: {aic}, BIC: {bic}")
    plot_classification_vs_price(features, hidden_states, item, hmm_model)

    return [aic, bic], hmm_model