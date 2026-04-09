#%%
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import  pandas as pd, numpy as np
from    matplotlib import pyplot as plt
from    statsmodels.tsa.vector_ar.vecm import coint_johansen
import  src.data_processing.feature_engineering as get
import  src.utils.plot_tools as myplot
import  src.data_processing.outlier_detection as outlier
from    src.utils.model_tools import item_name
from    data.bosstables import (barrows, kreearra, general_graardor, commander_zilyana,
                                kril_tsutsaroth, dagannoth_kings, nightmare, nex, corporeal_beast,
                                moons_of_peril, chambers_of_xeric, theatre_of_blood, tombs_of_amascut,
                                cerberus, zulrah, vorkath, grotesque_guardians, leviathan)
#%%
price_data = get.price_data
price_matrix_items, vol_matrix_items = get.item_data(price_data, True)
_, boss_matrix_items, boss_vol_matrix_items = get.boss_data()
#%%
item_groups = {
    "Barrows (Ahrim)":   [item_name("Ahrim's hood"), item_name("Ahrim's robetop"), item_name("Ahrim's robeskirt"), item_name("Ahrim's staff")],
    "Barrows (Dharok)":  [item_name("Dharok's helm"), item_name("Dharok's platebody"), item_name("Dharok's platelegs"), item_name("Dharok's greataxe")],
    "Barrows (Guthan)":  [item_name("Guthan's helm"), item_name("Guthan's platebody"), item_name("Guthan's chainskirt"), item_name("Guthan's warspear")],
    "Barrows (Karil)":   [item_name("Karil's coif"), item_name("Karil's leathertop"), item_name("Karil's leatherskirt"), item_name("Karil's crossbow")],
    "Barrows (Torag)":   [item_name("Torag's helm"), item_name("Torag's platebody"), item_name("Torag's platelegs"), item_name("Torag's hammers")],
    "Barrows (Verac)":   [item_name("Verac's helm"), item_name("Verac's brassard"), item_name("Verac's plateskirt"), item_name("Verac's flail")],
    "Barrows (All)":     barrows,
    "GWD Armadyl":       kreearra,
    "GWD Bandos":        general_graardor,
    "GWD Sara":          commander_zilyana,
    "GWD Zammy":         kril_tsutsaroth,
    "Dagannoth Kings":   dagannoth_kings,
    "Nightmare":         nightmare,
    "Nex":               nex,
    "Corp Beast":        corporeal_beast,
    "Cerberus":          cerberus,
    "Zulrah":            zulrah,
    "Vorkath":           vorkath,
    "Grotesque Guard.":  grotesque_guardians,
    "Leviathan":         leviathan,
    "Moons of Peril":    moons_of_peril,
    "Chambers of Xeric": chambers_of_xeric,
    "Theatre of Blood":  theatre_of_blood,
    "Tombs of Amascut":  tombs_of_amascut,
}
#%%
def johansen_test(data: pd.DataFrame, group_name: str, det_order: int = 0, k_ar_diff: int = 1, verbose: bool = True) -> dict|None:
    """
    Run the Johansen trace test on a group of price series.

    Parameters:
    - data: DataFrame with item price columns
    - group_name: label for printing
    - det_order: -1 (no const), 0 (constant), 1 (linear trend)
    - k_ar_diff: lagged differences in VAR
    """
    clean = data.dropna()
    if clean.shape[0] < 50:
        print(f"  Skipping {group_name}: insufficient observations ({clean.shape[0]})")
        return None
    if clean.shape[1] < 2:
        print(f"  Skipping {group_name}: fewer than 2 items")
        return None

    result = coint_johansen(clean, det_order, k_ar_diff)

    n_vars = clean.shape[1]
    n_coint = {90: 0, 95: 0, 99: 0}

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Johansen Trace Test: {group_name} ({n_vars} items)")
        print(f"  Observations: {clean.shape[0]}  |  det_order={det_order}  |  k_ar_diff={k_ar_diff}")
        print(f"{'='*70}")
        print(f"  {'H0':>6}  {'Trace Stat':>12}  {'90% CV':>10}  {'95% CV':>10}  {'99% CV':>10}  {'Result (95%)':>14}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")

    for i in range(n_vars):
        trace_stat = result.lr1[i]
        cv_90 = result.cvt[i, 0]
        cv_95 = result.cvt[i, 1]
        cv_99 = result.cvt[i, 2]
        reject_95 = trace_stat > cv_95

        if verbose:
            marker = "REJECT" if reject_95 else "fail to reject"
            print(f"  r <= {i:<3}  {trace_stat:>12.4f}  {cv_90:>10.4f}  {cv_95:>10.4f}  {cv_99:>10.4f}  {marker:>14}")

        if trace_stat > cv_90: n_coint[90] += 1
        if trace_stat > cv_95: n_coint[95] += 1
        if trace_stat > cv_99: n_coint[99] += 1

    if verbose:
        print(f"\n  Cointegrating relations:  90%: {n_coint[90]}  |  95%: {n_coint[95]}  |  99%: {n_coint[99]}")

    items = list(clean.columns)
    return {
        'group_name':    group_name,
        'items':         items,
        'trace_stats':   result.lr1,
        'crit_values':   result.cvt,
        'max_eig_stats': result.lr2,
        'max_eig_crit':  result.cvm,
        'eigenvalues':   result.eig,
        'eigenvectors':  result.evec,
        'n_coint_90':    n_coint[90],
        'n_coint_95':    n_coint[95],
        'n_coint_99':    n_coint[99],
    }
#%%
results = {}
for group_name, item_ids in item_groups.items():
    available = [item for item in item_ids if item in boss_matrix_items.columns]
    if len(available) < 2:
        print(f"Skipping {group_name}: fewer than 2 items available in data")
        continue

    group_data = boss_matrix_items[available].resample('h').mean()
    r = johansen_test(group_data, group_name, det_order=0, k_ar_diff=1)
    if r is not None:
        results[group_name] = r
#%%
summary = pd.DataFrame({
    'Group':              [r['group_name'] for r in results.values()],
    'N Items':            [len(r['items']) for r in results.values()],
    'Coint (90%)':        [r['n_coint_90'] for r in results.values()],
    'Coint (95%)':        [r['n_coint_95'] for r in results.values()],
    'Coint (99%)':        [r['n_coint_99'] for r in results.values()],
    'Max Eigenvalue':     [r['eigenvalues'][0] for r in results.values()],
})
print("\n\nSummary of Johansen Trace Tests")
print("=" * 80)
print(summary.to_string(index=False))
#%%
target_group = "Barrows (Dharok)"
r = results[target_group]
group_data = boss_matrix_items[r['items']].resample('h').mean().dropna()

coint_vector = r['eigenvectors'][:, 0]
spread = group_data.values @ coint_vector
spread_series = pd.Series(spread, index=group_data.index, name=f'{target_group} Spread')

myplot.plot_features(spread_series, title=f'{target_group} Cointegrating Spread (Eigenvector 1)')
#%%
window = 200
spread_mean = spread_series.rolling(window).mean()
spread_std = spread_series.rolling(window).std()
spread_z = ((spread_series - spread_mean) / spread_std).dropna()
spread_z.name = f'{target_group} Spread Z-Score'

fig, ax = myplot.plot_features(spread_z, title=f'{target_group} Spread Z-Score (window={window})')
ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='+2 SD')
ax.axhline(y=-2, color='green', linestyle='--', alpha=0.5, label='-2 SD')
ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
ax.legend()
#%%
for group_name, r in results.items():
    if r['n_coint_95'] > 0:
        group_data = boss_matrix_items[r['items']].resample('h').mean().dropna()
        coint_vector = r['eigenvectors'][:, 0]
        spread = group_data.values @ coint_vector
        spread_series = pd.Series(spread, index=group_data.index)

        spread_outliers = outlier.ewm_z_residuals(spread_series, 100, 3, 3)
        spread_filtered = spread_series.drop(spread_outliers.index)

        item_labels = ', '.join([str(item_name(i)) for i in r['items'][:3]])
        if len(r['items']) > 3:
            item_labels += '...'
        myplot.plot_features(spread_filtered, title=f'{group_name} Coint. Spread ({r["n_coint_95"]} relations at 95%)\n{item_labels}')
# %%
