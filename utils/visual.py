# %%
#Script Innit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: sys.path.append(project_root)
from   utils.model_tools import item_name, volatility_market
import utils.plot_tools as myplot
from   scipy.stats import norm, t, kurtosis, skew, jarque_bera, probplot, skewtest
import seaborn as sns
import utils.model_tools as tools
import testing.feature_engineering as get

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
# price_corr, vol_corr = get.item_corr(price_matrix_items, vol_matrix_items)
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
returns = tools.calculate_returns(item_price, "h")
log_returns = np.log(returns)
returns_volatility = log_returns.rolling(smoothing).std().dropna() 
#%%
myplot.plot_features(returns_volatility, ylab='Standard Deviations (GP)', title=fr'$\mathbf{{{item_name(item)}}}$ [{item}] Volatility')
#%% Leverage Effect (Returns vs Volatility)
plt.figure(figsize=(10,5))
plt.title(fr'$\mathbf{{{item_name(item)}}}$ [{item}] Volatility vs. Log Returns')
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
#219, 12934, 
myplot.plot_historical_alch_vs_price(item1)
myplot.plot_recent_alch_vs_price(item1)
#%% Log Return Distributional Visual

# aggregation is for modeling (low samples), while overlapping periods violate i.i.d 
# but autocorrelation won't change *visual* distribution 
return_period=140 #288 daily
#aggregation to larger timeframe vs. lagged (and overlapping) periods for >1
log_returns=np.log(1+item_price.pct_change(return_period).dropna())
#log_returns=np.log(1+price_matrix_items[item_price].resample('D').last().dropna()) #daily aggregation
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
