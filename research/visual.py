# %%
#Script Innit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: sys.path.append(project_root)
from   src.utils.model_tools import item_name, volatility_market
import src.utils.plot_tools as myplot
from   scipy.stats import norm, t, kurtosis, skew, jarque_bera, probplot, skewtest
import seaborn as sns
import src.utils.model_tools as tools
import src.data_processing.feature_engineering as get

plt.rcParams.update({
    "figure.facecolor": "#000000",  
    "axes.facecolor": "#000000",   
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "gray",
    "legend.facecolor": "#303030",
    "legend.labelcolor": "#A1A1A1",
    "axes.titlecolor": "#A1A1A1"
})
#%%
price_data = get.price_data
price_matrix_items, vol_matrix_items = get.item_data(price_data, True)
price_matrix_items = price_matrix_items.iloc[1000:30000]
vol_matrix_items = vol_matrix_items.iloc[1000:30000]

equal_index, vprice_index = get.market_indices(price_matrix_items, vol_matrix_items)
_, boss_matrix_items, boss_vol_matrix_items = get.boss_data()
# %%
#Price and Volume Time series
totalvolume_time = vol_matrix_items.iloc[:,1:].sum(axis=1)

item1 = np.random.choice(price_matrix_items.columns)
item2 = np.random.choice(price_matrix_items.columns)
item= np.random.choice(price_matrix_items.columns)

start = 20
market_volatility = volatility_market(price_matrix_items, smoothing=start, aggregation="h")
#%% Market Volatility Plot
myplot.plot_features(market_volatility, title=fr"OSRS $\mathbf{{{round((market_volatility.shape[0]*5)/(60*24),1)}}}$ Day Market Volatility", ylab="Standard Deviation (SD)")
#%% Item Volatility 
smoothing = 5
item_price= price_matrix_items[item]
returns = tools.calculate_returns(item_price,"h")
log_returns = np.log(1+returns)
returns_volatility = returns.rolling(smoothing).std().dropna() 
log_returns_volatility = log_returns.rolling(smoothing).std().dropna() 
#%%
myplot.plot_features(log_returns_volatility, ylab='Standard Deviations (GP)', title=fr'$\mathbf{{{item_name(item)}}}$ [{item}] Volatility')
#%% Leverage Effect (Returns vs Volatility)
plt.figure(figsize=(10,5))
plt.title(fr'$\mathbf{{{item_name(item)}}}$ [{item}] Log Return Volatility vs. Log Returns')
plt.scatter(returns_volatility, returns[smoothing-1:], alpha=0.5, color='skyblue', edgecolor='black')
plt.xlabel('Market Volatility')
plt.ylabel('Log Returns')
plt.grid()
plt.show()

#%% Total Volume Plot 
myplot.plot_features(totalvolume_time, ylab="Volume of Market Trade", title=fr"$\mathbf{{{round((market_volatility.shape[0]*5)/(60*24),1)}}}$ Day Market Volume of the $\mathbf{{{vol_matrix_items.shape[1]}}}$ Most Traded Items")
#%% Item Volume
myplot.plot_features(vol_matrix_items[item], ylab='Volume',title=fr"$\mathbf{{{item_name(item)}}}$ [{item}] $\mathbf{{{round((market_volatility.shape[0]*5)/(60*24),1)}}}$ Day Market Volume")

#%% Item Price vs Alchemy Price Plot

myplot.plot_historical_alch_vs_price(np.random.choice(price_matrix_items.columns))
myplot.plot_recent_alch_vs_price(np.random.choice(price_matrix_items.columns))
#%%
lower_bound = np.percentile(log_returns, 0.1)
upper_bound = np.percentile(log_returns, 99.9)

plt.figure(figsize=(10,5))
plt.xlabel('Log Returns')
plt.ylabel('Returns Density')
plt.xlim(lower_bound, upper_bound)
plt.axvline(0, color='white', linestyle='-', linewidth=1)
plt.yscale('log')
plt.title(fr"$\mathbf{{{item_name(item)}}}$ [{item}] Log Return Distribution") #ignore period for aggregated (?)


plt.hist(log_returns, bins='fd', color='skyblue', edgecolor='black', alpha=0.7, density=True)
ax=plt.gca()

mu_norm, std_norm = norm.fit(log_returns)
x_plot = np.linspace(lower_bound, upper_bound, 500)
pdf_norm = norm.pdf(x_plot, mu_norm, std_norm)
ax.plot(x_plot, pdf_norm, 'r-', linewidth=1, label=r'Normal ($H_0$)')

df_t, loc_t, scale_t = t.fit(log_returns)
pdf_t= t.pdf(x_plot,df_t,loc_t,scale_t)
ax.plot(x_plot, pdf_t, 'g-', linewidth=1, label='Student\'s t')
#Info Box

jb_stat, jb_p_value= jarque_bera(log_returns)
#significance test of skew (Z-test 2 tail)
skew_z_score, skew_p_value = skewtest(log_returns)

textstr = '\n'.join((
    r'$\mu$: %.4f' % (mu_norm,),
    r'$\sigma$: %.4f' % (std_norm,),
    r'Skewness: %.3f (p=%.3f)' % (skew(log_returns), skew_p_value),
    r'Kurtosis: %.3f' % (kurtosis(log_returns),),
    r'Jaque-Bera: %.3f (p=%.3f)' % (jb_stat,jb_p_value),
))
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left', bbox=props)

plt.legend()
plt.show()
 
#%% QQ Plot Normal
plt.figure(figsize=(10,5))
out=probplot(log_returns, dist="norm", fit=True, rvalue=True, plot=plt)

plt.text(0.1, 0.7, f'$R^2$={out[1][2]**2:.3f}', transform=plt.gca().transAxes, fontsize=13,color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
plt.title(fr'QQ Plot: $\mathbf{{{item_name(item)}}}$ [{item}] $\mathbf{{{return_period}}}$-Period Log Returns vs Normal')
plt.ylabel('Sample Quantiles')
plt.xlabel('Theoretical Quantiles')
plt.grid()
plt.show()

#%% QQ Plot t
plt.figure(figsize=(10,5))
out=probplot(log_returns, dist="t", sparams=(4.39,), fit=True, rvalue=True, plot=plt)

plt.text(0.1, 0.7, f'$R^2$={out[1][2]**2:.3f}', transform=plt.gca().transAxes, fontsize=13,color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
plt.title(fr'QQ Plot: $\mathbf{{{item_name(item)}}}$ [{item}] $\mathbf{{{return_period}}}$-Period Log Returns vs Student\'s t')
plt.ylabel('Sample Quantiles')
plt.xlabel('Theoretical Quantiles')
plt.grid()
plt.show()
# %%
myplot.plot_features(price_matrix_items[item1],start= '2025-05-10',end= '2025-05-11')
#%% Market Indices
myplot.plot_features(equal_index)
#%%
myplot.plot_features(equal_index)
# %% Volume History 
myplot.plot_features(vol_matrix_items[item1],start= '2025-05-10',end= '2025-05-11')
# %% Correlations
sns.heatmap(price_matrix_items.iloc[:,1:10].corr(), annot=True, cmap='coolwarm', fmt=".2f")
#%%
sns.heatmap(vol_matrix_items.iloc[:,15:30].corr(), annot=True, cmap='coolwarm', fmt=".2f")
#%% Item Beta
myplot.plot_item_market_divergence(price_matrix_items, item1, vprice_index)
#%%
myplot.plot_item_market_divergence(price_matrix_items, item1, equal_index)
print(f'Beta: {tools.beta(price_matrix_items, item1, equal_index)}')
#%% Pair Divergence
myplot.plot_feature_divergence(price_matrix_items[item1], price_matrix_items[item2], 'z', window=60,start= '2025-05-10',end= '2025-05-11')
#%%
high_value_items = price_matrix_items.loc[:,price_matrix_items.mean()>300]
high_value_volume = vol_matrix_items.loc[:,price_matrix_items.mean()>300]
#%%
market_index = tools.create_item_index(
        high_value_items,
        list(high_value_items.columns),
        type='equal',base_value=100)
market_index_v = tools.create_item_index(
        [high_value_items,high_value_volume],
        list(high_value_items.columns),
        type='vprice',base_value=100)
#%% Market Returns
market_returns = tools.calculate_returns(market_index, '5m').dropna()
log_market = np.log(1+market_returns)
lower_bound = np.percentile(log_market, 0.1)
upper_bound = np.percentile(log_market, 99.9)
#%%
plt.figure(figsize=(10,5))
plt.xlabel('Log Returns')
plt.ylabel('Returns Density')
plt.xlim(lower_bound, upper_bound)
plt.axvline(0, color='white', linestyle='-', linewidth=1)
plt.yscale('log')
plt.title(fr"Daily Market Log Return Distribution") #ignore period for aggregated (?)


plt.hist(log_market, bins='fd', color='skyblue', edgecolor='black', alpha=0.7, density=True)
ax=plt.gca()

mu_norm, std_norm = norm.fit(log_market)
x_plot = np.linspace(lower_bound, upper_bound, 500)
pdf_norm = norm.pdf(x_plot, mu_norm, std_norm)
ax.plot(x_plot, pdf_norm, 'r-', linewidth=1, label=r'Normal ($H_0$)')

df_t, loc_t, scale_t = t.fit(log_market)
pdf_t= t.pdf(x_plot,df_t,loc_t,scale_t)
ax.plot(x_plot, pdf_t, 'g-', linewidth=1, label='Student\'s t')
#Info Box

jb_stat, jb_p_value= jarque_bera(log_market)
#significance test of skew (Z-test 2 tail)
skew_z_score, skew_p_value = skewtest(log_market)

textstr = '\n'.join((
    r'$\mu$: %.4f' % (mu_norm,),
    r'$\sigma$: %.4f' % (std_norm,),
    r'Skewness: %.3f (p=%.3f)' % (skew(log_market), skew_p_value),
    r'Kurtosis: %.3f' % (kurtosis(log_market),),
    r'Jaque-Bera: %.3f (p=%.3f)' % (jb_stat,jb_p_value),
))
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left', bbox=props)

plt.legend()
plt.show()
# %% --------------------------------------

market_index_full = tools.create_item_index(
        price_matrix_items,
        list(price_matrix_items.columns),
        type='equal',base_value=100)
market_index_v_full = tools.create_item_index(
        [price_matrix_items,vol_matrix_items],
        list(price_matrix_items.columns),
        type='vprice',base_value=100)

market_returns_full = tools.calculate_returns(market_index_full, '5m').dropna()
log_market_full = np.log(1+market_returns_full)

alt_returns = market_index_full/market_index_full.shift(1)
alt_log_returns = np.log(alt_returns)
alt_log_returns_hour = alt_log_returns.resample("h").sum() 
#%% ACF
plt.figure(figsize=(10, 5))
plot_acf(alt_log_returns_hour, lags=range(20), alpha=0.05)
plt.ylim(-0.50,0.25)
plt.title("Market Return Autocorrelation")
plt.show()
#%% Spectral analysis
price_matrix_returns = price_matrix_items.pct_change()
price_matrix_log_returns = np.log(1+price_matrix_returns).iloc[1:]
#%%
log_item_corr = price_matrix_log_returns.corr()

import src.data_processing.r_pca as r_pca
daily_rpca = r_pca.R_pca(log_item_corr.values)
L, S = daily_rpca.fit(max_iter=10000, iter_print=100)

evalues, evectors = np.linalg.eigh(L)
idx = np.argsort(evalues)[::-1]
evalues_sort = evalues[idx]
evectors_sort = evectors[:, idx]
#block permutation test
#%%
from tqdm import tqdm
data_array = price_matrix_log_returns.values 
n_rows, n_cols = data_array.shape
block_size = 48 #4 hour period
permutations = 10
indices = np.arange(0, n_rows, block_size)
plt.figure(figsize=(10, 5))
for i in tqdm(range(permutations), desc="Permutations"):
    shuffled_matrix = np.zeros_like(data_array)
    
    for col in range(n_cols):
        shuffled_idx = np.random.permutation(indices)
        
        blocks = [data_array[start : start + block_size, col] for start in shuffled_idx]
        shuffled_matrix[:, col] = np.concatenate(blocks)[:n_rows] 
    print(f"Any NaNs in shuffled_matrix: {np.isnan(shuffled_matrix).any()}")
    stds = np.std(shuffled_matrix, axis=0)
    zero_std_count = np.sum(stds == 0)
    if zero_std_count > 0:
        print(f"Warning: {zero_std_count} columns have zero variance. This will cause NaNs in correlation.")
    shuffled_corr = np.corrcoef(shuffled_matrix, rowvar=False)
    shuffled_evals = np.linalg.eigvalsh(shuffled_corr)
    shuffled_evals = shuffled_evals[::-1] # Sort descending
    
    plt.plot(shuffled_evals, 'o-', markersize=2, color="red", alpha=0.3)
plt.plot(evalues_sort, 'o-', markersize=4)
plt.title("Market Spectrum: Signal vs. Noise (Optimized)")
plt.xlim(-1,100)
plt.legend()
plt.show()

# %% rotation
from factor_analyzer import Rotator
keep = 8

loading_matrix = evectors_sort
rotator = Rotator(method='promax')
rotated_loadings_array = rotator.fit_transform(loading_matrix[:,:keep])

for i in range(keep):
    component = i
    vector = rotated_loadings_array[:, component-1]
    v_series = pd.Series(vector, index=price_matrix_items.columns)
    top_n_index = v_series.abs().sort_values(ascending=False).head(10).index
    top_n = v_series.loc[top_n_index]
    print(f"Loadings for Factor {component+1  }:")
    name_list = pd.Series([item_name(iter) for iter in top_n.index], index=top_n_index)

    print(pd.concat([top_n,name_list], axis=1))
#%%
import networkx as nx

G = nx.Graph()
for i in range(10):
    
    factor_indices = np.argsort(np.abs(evectors_sort[:, i]))[::-1][:5]
    names = {idx: item_name(price_matrix_items.columns[idx]) for idx in factor_indices}

    for a in range(len(factor_indices)):
        for b in range(a + 1, len(factor_indices)):
            idx_a = factor_indices[a]
            idx_b = factor_indices[b]
            node_a = names[idx_a]
            node_b = names[idx_b]
            same_sign = np.sign(evectors_sort[idx_a, i]) == np.sign(evectors_sort[idx_b, i])
            if G.has_edge(node_a, node_b):
                G[node_a][node_b]['weight'] += evalues_sort[i]
            else:
                G.add_edge(node_a, node_b, weight=evalues_sort[i], 
                           sign='positive' if same_sign else 'negative')
edge_colors = ['blue' if G[u][v]['sign'] == 'positive' else 'red' for u, v in G.edges()]
plt.figure(figsize=(12, 12))

#k = spacing
pos = nx.spring_layout(G, k=1, iterations=50)
nx.draw(G, pos, 
        with_labels=True, 
        edge_color=edge_colors, 
        node_color='lightblue', 
        node_size=500,
        font_size=8,
        width=[G[u][v]['weight'] * 0.5 for u, v in G.edges()]) 

plt.title("OSRS Market Factor Network (Top 10 Factors)")
plt.show()
#%%
G_clean = G.copy()

edge_threshold = 1.4
low_weight_edges = [(u, v) for u, v, d in G_clean.edges(data=True) if d['weight'] < edge_threshold]
G_clean.remove_edges_from(low_weight_edges)
edge_colors_clean = []
for u, v, data in G_clean.edges(data=True):
    if data['sign'] == 'positive':
        edge_colors_clean.append('royalblue')
    else:
        edge_colors_clean.append('crimson')
G_clean.remove_nodes_from(list(nx.isolates(G_clean)))
nx.draw(G_clean, pos, 
        with_labels=True, 
        edge_color=edge_colors_clean, 
        node_color='lightblue', 
        node_size=500,
        font_size=8,
        width=[G[u][v]['weight'] * 0.5 for u, v in G.edges()])
plt.title("OSRS Market Factor Network (Top 10 Factors)")
plt.show()

#%% aggregated data, repeated

daily_price_matrix = pd.read_csv("../data/historical_price_matrix.csv", index_col="timestamp")
daily_volume_matrix = pd.read_csv("../data/historical_volume_matrix.csv", index_col="timestamp")

daily_high_value_items = daily_price_matrix.loc[:,daily_price_matrix.mean()>300]
daily_high_value_volume = daily_volume_matrix.loc[:,daily_price_matrix.mean()>300]
#%%
daily_market_index = tools.create_item_index(
        daily_high_value_items,
        list(daily_high_value_items.columns),
        type='equal',base_value=100)
daily_market_index_v = tools.create_item_index(
        [daily_high_value_items,daily_high_value_volume],
        list(daily_high_value_items.columns),
        type='vprice',base_value=100)
#%% Market Returns
daily_market_returns = tools.calculate_returns(daily_market_index).dropna()
daily_log_market = np.log(1+daily_market_returns)
daily_lower_bound = np.percentile(daily_log_market, 0.1)
daily_upper_bound = np.percentile(daily_log_market, 99.9)
#%%
plt.figure(figsize=(10,5))
plt.xlabel('Daily Log Returns')
plt.ylabel('Returns Density')
plt.xlim(daily_lower_bound, daily_upper_bound)
plt.axvline(0, color='white', linestyle='-', linewidth=1)
plt.yscale('log')
plt.title(fr"Daily Market Log Return Distribution") 

plt.hist(daily_log_market, bins='fd', color='skyblue', edgecolor='black', alpha=0.7, density=True)
ax=plt.gca()

mu_norm, std_norm = norm.fit(daily_log_market)
x_plot = np.linspace(daily_lower_bound, daily_upper_bound, 500)
pdf_norm = norm.pdf(x_plot, mu_norm, std_norm)
ax.plot(x_plot, pdf_norm, 'r-', linewidth=1, label=r'Normal ($H_0$)')

df_t, loc_t, scale_t = t.fit(daily_log_market)
pdf_t= t.pdf(x_plot,df_t,loc_t,scale_t)
ax.plot(x_plot, pdf_t, 'g-', linewidth=1, label='Student\'s t')
#Info Box

jb_stat, jb_p_value= jarque_bera(daily_log_market)
#significance test of skew (Z-test 2 tail)
skew_z_score, skew_p_value = skewtest(daily_log_market)

textstr = '\n'.join((
    r'$\mu$: %.4f' % (mu_norm,),
    r'$\sigma$: %.4f' % (std_norm,),
    r'Skewness: %.3f (p=%.3f)' % (skew(daily_log_market), skew_p_value),
    r'Kurtosis: %.3f' % (kurtosis(daily_log_market),),
    r'Jaque-Bera: %.3f (p=%.3f)' % (jb_stat,jb_p_value),
))
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left', bbox=props)
plt.legend()
plt.show()
# %%
daily_market_index_full = tools.create_item_index(
        daily_price_matrix,
        list(daily_price_matrix.columns),
        type='equal',base_value=100)
daily_market_index_v_full = tools.create_item_index(
        [daily_price_matrix,daily_volume_matrix],
        list(daily_price_matrix.columns),
        type='vprice',base_value=100)

daily_market_returns_full = tools.calculate_returns(daily_market_index_full)
daily_log_market_full = np.log(1+daily_market_returns_full)
#%% Spectral analysis
daily_price_matrix_returns = daily_price_matrix.pct_change()
daily_price_matrix_log_returns = np.log(1+daily_price_matrix_returns)
#%%
daily_log_item_corr = daily_price_matrix_log_returns.corr()
#robust PCA
import src.data_processing.r_pca as r_pca
daily_rpca = r_pca.R_pca(daily_log_item_corr.values)
L, S = daily_rpca.fit(max_iter=10000, iter_print=100)
#%%
robust_evalues, robust_evectors = np.linalg.eigh(L)
robust_idx = np.argsort(robust_evalues)[::-1]
robust_evalues_sort = robust_evalues[robust_idx]
robust_evectors_sort = robust_evectors[:, robust_idx]
#permutation test, runs after getting robust matrix
plt.figure(figsize=(10, 5))
for i in range(10):
    daily_log_price_matrix_shuffled = daily_price_matrix_log_returns.copy()
    for column in daily_log_price_matrix_shuffled.columns:
        daily_log_price_matrix_shuffled[column] = np.random.permutation(daily_log_price_matrix_shuffled[column].values)
    shuffled_corr = daily_log_price_matrix_shuffled.corr()
    shuffled_evals = np.linalg.eigvalsh(shuffled_corr)
    shuffled_evals = sorted(shuffled_evals, reverse=True)
    plt.plot(shuffled_evals, 'o-', markersize=2, color="red", alpha=0.5)
plt.plot(robust_evalues_sort, 'o-', markersize=4)
plt.title("Market Spectrum: Signal vs. Noise (Robust)")
plt.xlabel("Factor Number")
plt.xlim(-1,100)
plt.ylabel("Eigenvalue")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()
# %% rotation
keep = 5

loading_matrix = robust_evectors_sort
#rotator = Rotator(method='promax')
#rotated_loadings_array = rotator.fit_transform(loading_matrix[:,:keep])

for i in range(keep):
    component = i+1
    vector = loading_matrix[:, component-1]
    v_series = pd.Series(vector, index=daily_price_matrix.columns)
    top_n_index = v_series.abs().sort_values(ascending=False).head(10).index
    top_n = v_series.loc[top_n_index]
    print(f"Loadings for Factor {component}:")
    name_list = pd.Series([item_name(int(iter)) for iter in top_n.index], index=top_n_index)
    loadings = pd.concat([top_n,name_list], axis=1)
    print(loadings)

#%%
import networkx as nx

G = nx.Graph()
for i in range(10):
    
    factor_indices = np.argsort(np.abs(robust_evectors_sort[:, i]))[::-1][:5]
    names = {idx: item_name(int(daily_price_matrix.columns[idx])) for idx in factor_indices}

    for a in range(len(factor_indices)):
        for b in range(a + 1, len(factor_indices)):
            idx_a = factor_indices[a]
            idx_b = factor_indices[b]
            node_a = names[idx_a]
            node_b = names[idx_b]
            same_sign = np.sign(robust_evectors_sort[idx_a, i]) == np.sign(robust_evectors_sort[idx_b, i])
            if G.has_edge(node_a, node_b):
                G[node_a][node_b]['weight'] += robust_evalues_sort[i]
            else:
                G.add_edge(node_a, node_b, weight=robust_evalues_sort[i], 
                           sign='positive' if same_sign else 'negative')
edge_colors = ['blue' if G[u][v]['sign'] == 'positive' else 'red' for u, v in G.edges()]
plt.figure(figsize=(12, 12))

#k = spacing
pos = nx.spring_layout(G, k=1, iterations=50)
nx.draw(G, pos, 
        with_labels=True, 
        edge_color=edge_colors, 
        node_color='lightblue', 
        node_size=500,
        font_size=8,
        width=[G[u][v]['weight'] * 0.5 for u, v in G.edges()]) 

plt.title("OSRS Market Factor Network (Top 10 Factors)")
plt.show()
#%%
G_clean = G.copy()

pos = nx.spring_layout(G, k=5, iterations=150)
edge_threshold = 4
low_weight_edges = [(u, v) for u, v, d in G_clean.edges(data=True) if d['weight'] < edge_threshold]
G_clean.remove_edges_from(low_weight_edges)
edge_colors_clean = []
for u, v, data in G_clean.edges(data=True):
    if data['sign'] == 'positive':
        edge_colors_clean.append('royalblue')
    else:
        edge_colors_clean.append('crimson')
G_clean.remove_nodes_from(list(nx.isolates(G_clean)))
nx.draw(G_clean, pos, 
        with_labels=True, 
        edge_color=edge_colors_clean, 
        node_color='lightblue', 
        node_size=500,
        font_size=8,
        width=[G[u][v]['weight'] * 0.5 for u, v in G.edges()]) 
plt.title("OSRS Market Factor Network (Top 10 Factors)")
plt.show()
#%%