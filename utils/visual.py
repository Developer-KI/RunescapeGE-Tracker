# %%
#Script Innit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: sys.path.append(project_root)
from   utils.data_pipeline import data_preprocess2, volatility_market, alchemy_preprocess
from   utils.model_tools import item_name
import utils.plot_tools as myplot
from   scipy.stats import norm, t, kurtosis, skew, shapiro, jarque_bera, probplot
import seaborn as sns
from testing.feature_engineering import market_index_equal_weight as EQUAL_INDEX,   market_index_volume_weight as VPRICE_INDEX
import utils.model_tools as tools


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

# %%
#Read/Write processed data
price_data = data_preprocess2(read=True)
reference = alchemy_preprocess(read=True)

# %%
#Price and Volume Time series
price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")
volume_matrix_items = price_data.pivot(index='timestamp', columns='item_id', values='totalvol')
totalvolume_time = volume_matrix_items.iloc[:,1:].sum(axis=1)
corr_price_items = price_matrix_items.corr()
corr_volume_items = volume_matrix_items.corr()

target_item1 = np.random.choice(price_matrix_items.columns)
target_item2 = np.random.choice(price_matrix_items.columns)
start = 20
market_volatility = volatility_market(price_data, smoothing=start)
vix_index= market_volatility.iloc[start:].index
plot_index = pd.to_datetime(market_volatility.iloc[:].index, unit='s')

#%% Market Volatility Plot
plt.figure(figsize=(10, 5))
plt.plot(pd.to_datetime(vix_index, unit='s'), market_volatility.iloc[start:], marker="o", markersize='2', linestyle="-", label="Standard Deviation")

plt.ylabel("Standard Deviation (SD)")
plt.title(fr"OSRS $\mathbf{{{round((market_volatility.shape[0]*5)/(60*24),1)}}}$ Day Market Volatility")
plt.xticks(rotation=45)
plt.grid()
plt.show()

#%% Item Volatility 
item=np.random.choice(price_matrix_items.columns)
lookback=return_period=400 #288 daily
price_matrix_items.index= pd.to_datetime(price_matrix_items.index,unit='s')
item_price= price_matrix_items[item]
log_returns=np.log(1+item_price.pct_change(return_period).dropna())

volatility = price_matrix_items[item].rolling(return_period).std().dropna()[1+lookback:] #dropped first to match pct_change of log_returns + lookback shift
volatility.index= pd.to_datetime(volatility.index,unit='s')

plt.figure(figsize=(10,5))
plt.plot(volatility.index,volatility)
plt.xticks(rotation=45)
plt.title(fr'$\mathbf{{{item_name(item)}}}$ [{item}] $\mathbf{{{return_period}}}$-Period Volatility')
plt.ylabel('Standard Deviations (GP)')
plt.grid()
plt.show()
#%% Leverage Effect (Returns vs Volatility)
plt.figure(figsize=(10,5))
plt.title(fr'$\mathbf{{{item_name(item)}}}$ [{item}] $\mathbf{{{return_period}}}$-Period Volatility vs. $\mathbf{{{return_period}}}$-Period Lagged $\mathbf{{{lookback}}}$ Returns')
plt.scatter(volatility, log_returns[volatility.index].shift(lookback), alpha=0.5, color='skyblue', edgecolor='black')
plt.xlabel('Market Volatility')
plt.ylabel('Log Returns')
plt.grid()
plt.show()

#%% Total Volume Plot 
plt.figure(figsize=(10,5))
plt.plot(plot_index, totalvolume_time, marker="o", markersize='2', linestyle="-", label="Volume")
plt.ylabel("Volume of Market Trade")
plt.title(fr"$\mathbf{{{round((market_volatility.shape[0]*5)/(60*24),1)}}}$ Day Market Volume of the $\mathbf{{{volume_matrix_items.shape[1]}}}$ Most Traded Items")
plt.xticks(rotation=45)
plt.grid()
plt.show()

#%% Item Volume
item = 4087
plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(plot_index, unit='s'), volume_matrix_items[item], marker="o", markersize='2', linestyle="-", label="Volume")
plt.ylabel("Volume of Trade")
plt.title(fr"$\mathbf{{{item_name(item)}}}$ [{item}] $\mathbf{{{round((market_volatility.shape[0]*5)/(60*24),1)}}}$ Day Market Volume")
plt.xticks(rotation=45)
plt.grid()
plt.show()

#%% Item Price vs Alchemy Price Plot
#219, 12934, 
myplot.plot_historical_alch_vs_price(5304)
myplot.plot_recent_alch_vs_price(5304)
# %%
# Total Volume Plot 
plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(plot_index, unit='s'), totalvolume_time, marker="o", markersize='2', linestyle="-", label="Volume")
plt.ylabel("Volume of Market Trade")
plt.title(fr"$\mathbf{{{round((market_volatility.shape[0]*5)/(60*24),1)}}}$ Day Market Volume of the $\mathbf{{{volume_matrix_items.shape[1]}}}$ Most Traded Items")
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
plt.gca().ticklabel_format(style='plain', axis='y') 
plt.grid()
plt.show()
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
plt.title(fr"$\mathbf{{{item_name(item)}}}$ [{item}] $\mathbf{{{return_period}}}$-Period Log Return Distribution") #ignore period for aggregated (?)


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
shapiro_stat, shapiro_p_value = shapiro(log_returns)

textstr = '\n'.join((
    r'$\mu$: %.4f' % (mu_norm,),
    r'$\sigma$: %.4f' % (std_norm,),
    r'Skewness: %.3f' % (skew(log_returns),),
    r'Kurtosis: %.3f' % (kurtosis(log_returns),),
    r'Jaque-Bera: %.3f (p=%.3f)' % (jb_stat,jb_p_value),
    r'Shapiro-Wilk: %.3f (p=%.3f)' % (shapiro_stat,shapiro_p_value)
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
myplot.plot_features(price_matrix_items[target_item1],start= '2025-05-10',end= '2025-05-11')
#%% Market Indices
myplot.plot_features(EQUAL_INDEX)
#%%
myplot.plot_features(VPRICE_INDEX)
# %% Volume History 
myplot.plot_features(volume_matrix_items[target_item1],start= '2025-05-10',end= '2025-05-11')
# %% Correlations
sns.heatmap(price_matrix_items.iloc[:,1:10].corr(), annot=True, cmap='coolwarm', fmt=".2f")
#%%
sns.heatmap(volume_matrix_items.iloc[:,15:30].corr(), annot=True, cmap='coolwarm', fmt=".2f")
#%% Item Beta
myplot.plot_item_market_divergence(price_matrix_items, 2, market_index_equal_weight)
#%%
myplot.plot_item_market_divergence(price_matrix_items, 2, market_index_equal_weight)
print(f'Beta: {tools.beta(price_matrix_items, target_item1, market_index_equal_weight)}')
#%% Pair Divergence
myplot.plot_feature_divergence(price_matrix_items[target_item1], price_matrix_items[target_item2], 'z', window=60,start= '2025-05-10',end= '2025-05-11')
#%%
