# %% Script Init
import pandas as pd
import numpy as np
import DataPipeline as pipeline
import ModelTools as tools
import Models.RFTS as myRFTS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import Models.HMM as myHMM
# %% Aggregate features for model
price_data = pipeline.data_preprocess(read=False, interp_method='nearest')
price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")
vol_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="totalvol")

target_item = 12934

price_items_reg = price_matrix_items[[target_item]].iloc[20:]
vol_items_reg = vol_matrix_items[[target_item]].iloc[20:]
price_items_reg.columns = [f'{target_item}']
vol_items_reg.columns = [f'{target_item}_vol']
volatility_index = pipeline.volatility_market(price_data, smoothing=20)[20:]
reg_data = pd.concat([price_items_reg, vol_items_reg, volatility_index], axis=1)

df_time = tools.target_time_features(reg_data, f'{target_item}', 10)
df_roll = tools.target_rolling_features(reg_data, f'{target_item}', 20)
df = pd.merge(df_time, df_roll, on='timestamp', how='inner').dropna()
df_mod=df[['12934','lag1','lag2','lag3','lag4']]
# %% Run the RFTS model
model, test_idx = myRFTS.RFTS(data=df_mod, target_col=f'{target_item}', n_estimators=200,time_splits=30)

#Plot RFTS predictions vs realized price
X = df_mod.drop(f'{target_item}', axis=1)
Y = df_mod[f'{target_item}']

tools.plot_pred_vs_price(Y.iloc[test_idx[:100]], X.iloc[test_idx[:100]], model=model)
# %%
myHMM.ItemThresholdHMM(price_matrix_items,207)
#%%

booleandf = tools.rolling_threshold_classification(price_matrix_items,100,0.1)
item=207

X=booleandf[item].values.reshape(-1,1)
X[0,0]=2
n_components=len(np.unique(X))
encoder = myHMM.OneHotEncoder(sparse_output=False, categories='auto')
X_encoded = encoder.fit_transform(X).astype(int)[:,0:2]  # Shape will now be (2499, 2), excluded a dummy for reference

iter = 10000
HMMmodel = myHMM.MultinomialHMM(n_components=n_components, n_iter=iter) #leave init_params empty to self-select probabilities
#HMMmodel.emissionprob_ = np.array(emissionprob)
HMMmodel.fit(X_encoded)
hidden_states= HMMmodel.predict(X_encoded)

log_likelihood = HMMmodel.score(X_encoded)
# Estimate number of parameters
num_parameters = n_components**2 +(n_components*X.shape[1]) + n_components
# Compute AIC & BIC
aic = 2 * num_parameters - 2 * log_likelihood
bic = num_parameters * np.log(X.shape[0]) - 2 * log_likelihood

print(f"AIC: {aic}, BIC: {bic}")
tools.plot_classification_vs_price(price_matrix_items,hidden_states,item,HMMmodel)
#%%
abc = pd.DataFrame(np.random.uniform(0, 100, size=(100, 6)), columns=list('ABCDEF'))
meow = tools.rolling_threshold_classification(abc, 10, 14)
print(meow)
#%%
newoptim = myRFTS.RFTSOptim(df_mod,target_col=f'{target_item}',n_trials=5)
#%%
Xoptim = df_mod.drop(f'{target_item}', axis=1)
Yoptim = df_mod[f'{target_item}']
optim_idx = newoptim['Indices']
tools.plot_pred_vs_price(Yoptim.iloc[optim_idx[:100]], Xoptim.iloc[optim_idx[:100]], model=newoptim['Best Model'])
#%%

#somemodel, test_idx = myRFTS.RFTS(data=df, target_col=f'{target_item}',max_features='log2', min_samples_split=7, n_estimators=34)
#depths = [tree.get_depth() for tree in somemodel.estimators_]
#print(f"Average depth: {sum(depths) / len(depths):.2f}")
#print(f"Max depth: {max(depths)}, Min depth: {min(depths)}")
