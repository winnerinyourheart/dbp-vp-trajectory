# -*- coding: utf-8 -*-
"""Figure S4 (revision) - continuous norepinephrine-equivalent dose sensitivity, MIMIC-IV.

Panels:
  A  Mean DBP by class across the eight 6-hour windows.
  B  Mean log-transformed NE-equivalent dose (winsorised at the 99th percentile)
     by class across the same windows.

Classes are refit exactly as in the revised pipeline (rev_analysis_v1.py section [5]):
  X = [8 DBP windows (forward+backward LOCF), 8 dose windows (already ln(AUC+eps))]
  -> StandardScaler -> GaussianMixture(K=6, full, seed 42, n_init 20) -> relabel by
     ascending 28-day mortality (C0 low ... C5 high).
Diagnostics printed on run must reproduce the manuscript legend: n = 4,869,
mortality range 2.9%-59.9%, class+SOFA C = 0.816, ARI vs binary partition = 0.385.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'submission_clep', 'Figures_rev')
os.makedirs(OUT, exist_ok=True)

D = [f'dbp_w{i}' for i in range(8)]
Dc = [f'dose_w{i}' for i in range(8)]


def fit_gmm(X, K, seed=42, n_init=20):
    Xs = StandardScaler().fit_transform(X)
    gm = GaussianMixture(n_components=K, covariance_type='full', random_state=seed,
                         n_init=n_init, max_iter=500, tol=1e-4).fit(Xs)
    return gm.predict(Xs)


def order_by_mort(cl, y, K):
    order = sorted(range(K), key=lambda c: (y[cl == c].mean() if (cl == c).sum() else 9))
    m = {o: n for n, o in enumerate(order)}
    return np.array([m[c] for c in cl])


mm = pd.read_csv(os.path.join(BASE, 'binary_mimic_data.csv'))
cd = pd.read_csv(os.path.join(BASE, 'cont_dose_v4_mimic_data.csv'))
cd = cd[cd['stay_id'].isin(mm['stay_id'])].reset_index(drop=True)

cdd = cd[D].ffill(axis=1).bfill(axis=1)
dose = cd[Dc].copy()
Xcd = np.column_stack([cdd.values, dose.values])
raw = fit_gmm(Xcd, 6)
yc = cd['mort_28d'].values.astype(int)
ccl = order_by_mort(raw, yc, 6)

sizes = [int((ccl == c).sum()) for c in range(6)]
morts = [float(yc[ccl == c].mean()) * 100 for c in range(6)]
print('S4 n            =', len(cd))
print('S4 class sizes  =', sizes)
print('S4 class mort%  =', [round(x, 1) for x in morts])
try:
    V = [f'vaso_w{i}' for i in range(8)]
    Xb = np.column_stack([mm[D].ffill(axis=1).bfill(axis=1).values, mm[V].values])
    clb = order_by_mort(fit_gmm(Xb, 6), mm['mort_28d'].values.astype(int), 6)
    _b = pd.DataFrame({'stay_id': mm['stay_id'].values, 'clb': clb})
    _j = cd[['stay_id']].assign(ccl=ccl).merge(_b, on='stay_id', how='inner')
    print('ARI vs binary   =', round(float(adjusted_rand_score(_j['clb'], _j['ccl'])), 3),
          f'(matched n={len(_j)})')
except Exception as e:
    print('ARI diagnostic skipped:', e)

# panel data (display)
dbp = cdd
dwin = dose.copy()
for c in Dc:                                      # winsorise at 99th percentile
    dwin[c] = dwin[c].clip(upper=dwin[c].quantile(0.99))

w = np.arange(8)
cmap = plt.get_cmap('viridis')
fig, ax = plt.subplots(1, 2, figsize=(7.9, 3.6))
handles, labels = [], []
for i in range(6):
    m = ccl == i
    l1, = ax[0].plot(w, dbp[m].mean(axis=0), marker='o', ms=3,
                     lw=2.2 if i >= 4 else 1.3, color=cmap(i / 5))
    ax[1].plot(w, dwin[m].mean(axis=0), marker='o', ms=3,
               lw=2.2 if i >= 4 else 1.3, color=cmap(i / 5))
    handles.append(l1)
    labels.append(f'C{i} (n={sizes[i]:,})')

for a, t in zip(ax, ['A', 'B']):
    a.set_title(t, loc='left', fontweight='bold', fontsize=10)
    a.set_xlabel('6-hour window')
    a.set_xticks(w)
    a.set_xticklabels([f'W{i}' for i in range(8)])
    a.grid(ls=':', alpha=.5)
ax[0].set_ylabel('Mean DBP (mmHg)')
ax[1].set_ylabel('Mean log NE-equivalent dose')

leg = fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, fontsize=7.5)
for t in leg.get_texts():                         # bold: two highest-mortality classes
    if t.get_text().startswith(('C4', 'C5')):
        t.set_fontweight('bold')

fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(os.path.join(OUT, 'Figure_S4.tif'), dpi=600, format='tiff',
            bbox_inches='tight', pad_inches=0.2, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig(os.path.join(OUT, 'Figure_S4.png'), dpi=300, format='png',
            bbox_inches='tight', pad_inches=0.2)
plt.close()
print('Saved Figure_S4.tif/.png ->', OUT)
