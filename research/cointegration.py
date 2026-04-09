#%%
import  os, sys
#For relative pathing
# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to reach the project root
project_root = os.path.join(current_dir, '..')
# Add the project root to the system path
if project_root not in sys.path:
    sys.path.append(project_root)

import  pandas as pd, numpy as np
from    matplotlib import pyplot as plt
from    matplotlib import ticker as mticker
import  src.data_processing.feature_engineering as get
import  src.utils.plot_tools as myplot
import  src.data_processing.outlier_detection as outlier
from    src.utils.model_tools import item_name
#%%
price_data = get.price_data
price_matrix_items, vol_matrix_items = get.item_data(price_data, False)

cointegration_pvalues = get.cointegration_pairs(price_matrix_items)
min_pvalue_mask = cointegration_pvalues[cointegration_pvalues['p_value']==0.0]

#%%
random_min_pvalue_pair = min_pvalue_mask.sample(n=1)
pair_item1 = random_min_pvalue_pair['item1'].iloc[0]
pair_item2 = random_min_pvalue_pair['item2'].iloc[0]
# pair_item2 = np.random.choice(np.unique(min_pvalue_mask['item1']))
pair_difference = np.abs(price_matrix_items[pair_item1]-price_matrix_items[pair_item2])
pair_difference_outliers = outlier.ewm(pair_difference, 100,1,1)
pair_difference_filtered = pair_difference.drop(pair_difference_outliers.index)
#%%
myplot.plot_features(pair_difference_filtered, start='2025-06-01', end='2025-06-28')
#%%
myplot.plot_features(pair_difference_filtered)
# %%
for row in min_pvalue_mask.index:
    item1 = min_pvalue_mask.loc[row,'item1']
    item2 = min_pvalue_mask.loc[row,'item2']
    pair_difference = np.abs(price_matrix_items[pair_item1]-price_matrix_items[pair_item2])
    pair_difference_outliers = outlier.ewm(pair_difference, 100,1,1)
    pair_difference_filtered = pair_difference.drop(pair_difference_outliers.index)
    myplot.plot_features(pair_difference_filtered, title=f'{item_name(item1)}: {item1} and {item_name(item2)}: {item2}')
#%%
r_path = os.path.join('C:\\','Users','Mstav','anaconda3','envs','runescape','Lib','R')
os.environ['R_HOME'] = r_path
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from src.data.data_pipeline import data_explicit_preprocess
from src.utils.model_tools import item_name
import statsmodels.api as sm
from itertools import combinations


print("WOW 1")

boss_data, boss_matrix_items, boss_vol_matrix_items = get.boss_data()
#%%
barrows = [
item_name("Ahrim's hood"),
item_name("Ahrim's robetop"),
item_name("Ahrim's robeskirt"),
item_name("Ahrim's staff"),
item_name("Dharok's helm"),
item_name("Dharok's platebody"),
item_name("Dharok's platelegs"),
item_name("Dharok's greataxe"),
item_name("Guthan's helm"),
item_name("Guthan's platebody"),
item_name("Guthan's chainskirt"),
item_name("Guthan's warspear"),
item_name("Karil's coif"),
item_name("Karil's leathertop"),
item_name("Karil's leatherskirt"),
item_name("Karil's crossbow"),
item_name("Torag's helm"),
item_name("Torag's platebody"),
item_name("Torag's platelegs"),
item_name("Torag's hammers"),
item_name("Verac's helm"),
item_name("Verac's brassard"),
item_name("Verac's plateskirt"),
item_name("Verac's flail"),
]
#ahrim_outliers = outlier.rolling_volume(boss_matrix_items[item_name("Ahrim's hood")], boss_vol_matrix_items[item_name("Ahrim's hood")], 30, 3)
#ahrim_filtered = boss_matrix_items[item_name("Ahrim's hood")].drop(ahrim_outliers.index)
#myplot.plot_features(ahrim_filtered)
#%%

# Unit root tests are not robust to heteroskedasticity (not a big deal)
# However, critical values cannot be generated for different model specifications and sample sizes
# Therefore, Lee Strazicich (among others) are invalid besides T=100, consider resampling
# Lee Strazicich null: Unit root *with* break | alt: Stationary *with* break 

r_utils_dir = os.path.join("C:", os.sep, "Users", "Mstav", "RunescapeGET", "RunescapeGET", "utils")
r_file1_path = os.path.join(r_utils_dir, 'LeeStrazicichUnitRootTest.R')
r_file2_path = os.path.join(r_utils_dir, 'LeeStrazicichUnitRootTestParallelization.R')

# Replace backslashes with forward slashes for R
r_file1_path_r = r_file1_path.replace(os.sep, '/')
r_file2_path_r = r_file2_path.replace(os.sep, '/')
ro.r(f'source("{r_file1_path_r}")')
ro.r(f'source("{r_file2_path_r}")')
ro.r('library("parallel")')
ro.r('library("doSNOW")')
ro.r('library("foreach")')

#%%
barrows_matrix_items = boss_matrix_items[barrows]
barrows_aggregate = barrows_matrix_items.resample('h').mean()
#%%
for item1, item2 in combinations(barrows_aggregate.columns, 2):
    #Generate residuals for unit-root test
    X = sm.add_constant(barrows_aggregate[item1])
    y = barrows_aggregate[item2]
    model = sm.OLS(y, X)
    results = model.fit()
    residuals = results.resid

    with (ro.default_converter + pandas2ri.converter).context():
        r_boss_resid = ro.conversion.get_conversion().py2rpy(residuals)
    
    ur_ls_bootstrap = ro.r['ur.ls.bootstrap']

    results_one_break = ur_ls_bootstrap(
    r_boss_resid,
    model = "crash",
    breaks = 1,
    method = "GTOS",
    pn = 0.1,
    print_results = "print"
    )



print("Done!")


# %%
