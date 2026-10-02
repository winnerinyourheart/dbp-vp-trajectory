"""
Train independent GMM on eICU and compare trajectory patterns with MIMIC
"""
import pandas as pd, numpy as np, pickle, os
from sklearn.mixture import GaussianMixture
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


BASE = DATA_DIR

# Load eICU thresholded features (0.01 threshold)
e = pd.read_csv(f'{BASE}/eicu_diag_features_cont.csv')
feat = [f'dbp_w{w}' for w in range(8)] + [f'vaso_w{w}' for w in range(8)]

# Fill NaN for DBP
for w in range(8):
    col = f'dbp_w{w}'
    e[col] = e[col].fillna(e[col].mean() if e[col].notna().any() else 60)

# Try multiple thresholds for vaso
for thr_label, thr in [('binary', None), ('thr001', 0.01), ('thr005', 0.05), ('thr010', 0.10)]:
    print(f'\n{"="*60}')
    print(f'=== eICU GMM with {thr_label} vaso definition ===')
    print(f'{"="*60}')
    
    e_work = e.copy()
    if thr is not None:
        for w in range(8):
            e_work[f'vaso_w{w}'] = (e_work[f'dose_w{w}'] > thr).astype(int)
    
    X = e_work[feat].fillna(0).values
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    # GMM K=6
    gmm = GaussianMixture(n_components=6, random_state=42, n_init=20)
    e_cl = gmm.fit_predict(Xs)
    
    # Reorder classes by mortality
    mort_by_cl = {c: e_work[e_cl == c]['mort'].mean() for c in range(6)}
    order = sorted(range(6), key=lambda c: mort_by_cl[c])
    cmap = {old: new for new, old in enumerate(order)}
    e_cl_re = np.array([cmap[c] for c in e_cl])
    
    e_work['class'] = e_cl_re
    
    print(f'Class mortality:')
    for c in range(6):
        sub = e_work[e_work['class'] == c]
        sofa_mean = sub['sofa'].fillna(sub['sofa'].mean()).mean()
        print(f'  C{c}: n={len(sub):4d}, mort={sub["mort"].mean():.1%}, SOFA={sofa_mean:.1f}')
    
    # Cox C-stat
    dm = pd.DataFrame({'surv': [28]*len(e_work), 'event': e_work['mort'].values})
    for c in range(1, 6):
        dm[f'c{c}'] = (e_work['class'] == c).astype(int)
    
    try:
        cu = CoxPHFitter(penalizer=0.01).fit(dm[['surv','event']+[f'c{c}' for c in range(1,6)]], 'surv', 'event')
        print(f'  Unadj C-stat: {cu.concordance_index_:.4f}')
        
        dm['sofa'] = e_work['sofa'].fillna(e_work['sofa'].mean()).values
        ca = CoxPHFitter(penalizer=0.01).fit(dm[['surv','event','sofa']+[f'c{c}' for c in range(1,6)]], 'surv', 'event')
        print(f'  +SOFA C-stat: {ca.concordance_index_:.4f}')
        
        # Show HR
        print(f'  Hazard ratios (unadj):')
        for c in range(1, 6):
            hr = np.exp(cu.params_[f'c{c}'])
            lo = np.exp(cu.confidence_intervals_.loc[f'c{c}', 'lower-bound'])
            hi = np.exp(cu.confidence_intervals_.loc[f'c{c}', 'upper-bound'])
            print(f'    C{c} vs C0: HR={hr:.2f} ({lo:.2f}-{hi:.2f})')
    except Exception as ex:
        print(f'  Cox failed: {ex}')

    # Save model
    pickle.dump({'gmm': gmm, 'scaler': scaler, 'class_map': cmap,
                  'data': e_work}, open(f'{BASE}/eicu_gmm_{thr_label}.pkl', 'wb'))

# Also load MIMIC for visual comparison
m = pd.read_csv(f'{BASE}/binary_mimic_data.csv')
print(f'\n{"="*60}')
print(f'=== MIMIC GMM reference (K=6 seed=42) ===')
print(f'{"="*60}')
for c in range(6):
    sub = m[m['class'] == c]
    print(f'  C{c}: n={len(sub):4d}, mort={sub["mort_28d"].mean():.1%}')
