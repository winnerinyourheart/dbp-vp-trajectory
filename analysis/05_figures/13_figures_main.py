# -*- coding: utf-8 -*-
"""
CLEP 返修：主图与补充图重建（数值已按统一 LOCF 主分析更新）
输出: Figures_rev/ 下的 300-600dpi TIFF + PNG 预览
设计约束（Dovepress）: 图内不放标题/说明性文字；粗体含义在图注说明；文字不重叠。
"""
import sys, os, json, warnings, pickle
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import adjusted_rand_score
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


BASE = DATA_DIR
OUT = os.path.join(BASE, "submission_clep", "Figures_rev")
os.makedirs(OUT, exist_ok=True)
D = [f"dbp_w{i}" for i in range(8)]
V = [f"vaso_w{i}" for i in range(8)]
SEED = 42
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "savefig.bbox": "tight"})
# 红=高风险, 蓝=低风险（按死亡风险递增的配色）
CMAP = ["#1f4e9c", "#3d7ebf", "#7fb3d9", "#f2b04a", "#e0732a", "#b02418"]
LBL = [f"C{i}" for i in range(6)]

def order_by_mort(cl, y, K):
    order = sorted(range(K), key=lambda c: (y[cl == c].mean() if (cl == c).sum() else 9))
    m = {o: n for n, o in enumerate(order)}
    return np.array([m[c] for c in cl])

def fit(X, K, seed=SEED, n_init=20):
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    gm = GaussianMixture(K, covariance_type="full", random_state=seed, n_init=n_init,
                         max_iter=500, tol=1e-4).fit(Xs)
    return gm, sc, gm.predict(Xs)

# ---------------- 数据 ----------------
m = pd.read_csv(os.path.join(BASE, "binary_mimic_data.csv"))
m["surv"] = m["surv"].clip(lower=0.01)
Db = m[D].ffill(axis=1).bfill(axis=1)
X = np.column_stack([Db.values, m[V].values])
gm, sc, cl = fit(X, 6)
y = m["mort_28d"].values.astype(int)
cl = order_by_mort(cl, y, 6)
m["cls"] = cl
surv = m["surv"].values.astype(float); sofa = m["sofa"].values.astype(float)

# ---------------- Figure 1: trajectories ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
w = np.arange(8)
for c in range(6):
    d = m[m.cls == c]
    ax[0].plot(w, d[D].mean().values, color=CMAP[c], marker="o", ms=3.2, lw=1.4, label=f"{LBL[c]} (n={len(d)})")
    ax[1].plot(w, d[V].mean().values * 100, color=CMAP[c], marker="o", ms=3.2, lw=1.4)
ax[0].set_xlabel("6-hour window (0\u201348 h from ICU admission)")
ax[0].set_ylabel("Mean DBP (mmHg)")
ax[1].set_xlabel("6-hour window (0\u201348 h from ICU admission)")
ax[1].set_ylabel("Patients on vasopressors (%)")
for a in ax:
    a.set_xticks(w); a.set_xticklabels([f"W{i}" for i in range(8)])
    a.grid(alpha=.25, lw=.6)
ax[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_1.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure_1.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig); print("[OK] Figure_1")

# ---------------- Figure 2: Kaplan-Meier ----------------
fig, a = plt.subplots(figsize=(4.4, 3.4))
for c in range(6):
    d = m[m.cls == c]
    kmf = KaplanMeierFitter().fit(d["surv"], d["mort_28d"])
    kmf.plot_survival_function(ax=a, ci_show=False, color=CMAP[c], lw=1.5, label=f"{LBL[c]} (n={len(d)})")
a.set_xlabel("Days from ICU admission"); a.set_ylabel("Survival probability")
a.set_xlim(0, 28); a.set_ylim(0.55, 1.005); a.grid(alpha=.25, lw=.6)
a.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_2.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure_2.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig); print("[OK] Figure_2")

# ---------------- Figure 3: forest plot ----------------
dd = pd.DataFrame({"time": surv, "event": y})
for c in range(1, 6):
    dd[f"c{c}"] = (cl == c).astype(int)
cp = CoxPHFitter(penalizer=0.0).fit(dd, "time", "event")
fig, a = plt.subplots(figsize=(4.6, 3.0))
ys, xs, los, his = [], [], [], []
for i, c in enumerate(range(1, 6)):
    r = cp.summary.loc[f"c{c}"]
    hr = float(np.exp(r["coef"])); lo = float(np.exp(r["coef"] - 1.96 * r["se(coef)"]))
    hi = float(np.exp(r["coef"] + 1.96 * r["se(coef)"]))
    ys.append(i); xs.append(hr); los.append(lo); his.append(hi)
ys = np.array(ys)
a.errorbar(xs, ys, xerr=[np.array(xs) - np.array(los), np.array(his) - np.array(xs)],
           fmt="o", ms=4.5, color="#1f4e9c", ecolor="#1f4e9c", capsize=2.5, lw=1.1)
a.axvline(1, color="grey", ls="--", lw=0.9)
a.set_yticks(ys); a.set_yticklabels([f"C{c} vs C0" for c in range(1, 6)])
a.set_xscale("log"); a.set_xlabel("Hazard ratio for 28-day mortality (log scale)")
a.grid(axis="x", alpha=.25, lw=.6)
for i, (xx, ll, hh) in enumerate(zip(xs, los, his)):
    a.annotate(f"{xx:.2f} ({ll:.2f}\u2013{hh:.2f})", (hh, i), xytext=(6, 0),
               textcoords="offset points", va="center", fontsize=7)
a.set_xlim(0.4, 12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_3.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure_3.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig); print("[OK] Figure_3")

# ---------------- eICU 数据 ----------------
ee = pickle.load(open(os.path.join(BASE, "eicu_gmm_binary.pkl"), "rb"))["data"].copy()
raw = pd.read_csv(os.path.join(BASE, "eicu_dbp_windows_raw.csv"))
ee = ee.drop(columns=D).merge(raw, on="pid", how="left")
ee = ee[ee[D].notna().sum(axis=1) >= 4].reset_index(drop=True)
ee["surv"] = ee["los_days"].clip(lower=0.01, upper=28.0)
ee["mort"] = ee["mort"].astype(int)
Xe = np.column_stack([ee[D].ffill(axis=1).bfill(axis=1).values, ee[V].values])
_, _, ecl = fit(Xe, 6)
ye = ee["mort"].values.astype(int)
ecl = order_by_mort(ecl, ye, 6)
ee["cls"] = ecl

# ---------------- Figure 4: eICU panels ----------------
fig, a = plt.subplots(2, 2, figsize=(7.2, 5.6))
mm_m = [y[cl == c].mean() * 100 for c in range(6)]
mm_e = [ye[ecl == c].mean() * 100 for c in range(6)]
a[0, 0].bar(range(6), mm_m, color=CMAP); a[0, 0].set_title("A  MIMIC-IV 28-day mortality", fontsize=8.5, loc="left")
a[0, 0].set_xticks(range(6)); a[0, 0].set_xticklabels(LBL); a[0, 0].set_ylabel("Mortality (%)")
a[0, 1].bar(range(6), mm_e, color=CMAP); a[0, 1].set_title("B  eICU-CRD ICU mortality", fontsize=8.5, loc="left")
a[0, 1].set_xticks(range(6)); a[0, 1].set_xticklabels(LBL); a[0, 1].set_ylabel("Mortality (%)")
for i in range(6):
    a[0, 0].annotate(f"{mm_m[i]:.1f}", (i, mm_m[i]), ha="center", va="bottom", fontsize=7)
    a[0, 1].annotate(f"{mm_e[i]:.1f}", (i, mm_e[i]), ha="center", va="bottom", fontsize=7)
sm = [m.sofa[cl == c].mean() for c in range(6)]
se = [ee.sofa[ecl == c].mean() for c in range(6)]
a[1, 0].plot(range(6), sm, "o-", color="#1f4e9c", lw=1.4, ms=4, label="MIMIC-IV")
a[1, 0].plot(range(6), se, "s--", color="#b02418", lw=1.4, ms=4, label="eICU-CRD")
a[1, 0].set_title("C  Baseline SOFA by class", fontsize=8.5, loc="left")
a[1, 0].set_xticks(range(6)); a[1, 0].set_xticklabels(LBL); a[1, 0].set_ylabel("Mean SOFA")
a[1, 0].legend(frameon=False, fontsize=7.5)
labs = ["Class\nonly", "Class\n+ SOFA"]
mmi = [cp.concordance_index_ if False else None, None]
def cstat(time, ev, Xdf, pen=0.05):
    d = pd.DataFrame({"time": time, "event": ev})
    for c2 in Xdf.columns: d[c2] = Xdf[c2].values
    return CoxPHFitter(penalizer=pen).fit(d, "time", "event").concordance_index_
Xc = pd.DataFrame({f"c{c}": (cl == c).astype(float) for c in range(1, 6)})
Xe2 = pd.DataFrame({f"c{c}": (ecl == c).astype(float) for c in range(1, 6)})
mm_c = [cstat(surv, y, Xc), cstat(surv, y, Xc.assign(sofa=sofa))]
ee_c = [cstat(ee.surv.values, ye, Xe2), cstat(ee.surv.values, ye, Xe2.assign(sofa=ee.sofa.fillna(ee.sofa.median()).values))]
xx = np.arange(2); wdt = 0.35
a[1, 1].bar(xx - wdt / 2, mm_c, wdt, color="#1f4e9c", label="MIMIC-IV")
a[1, 1].bar(xx + wdt / 2, ee_c, wdt, color="#b02418", label="eICU-CRD")
a[1, 1].set_xticks(xx); a[1, 1].set_xticklabels(labs)
a[1, 1].set_title("D  C-statistic by model", fontsize=8.5, loc="left")
a[1, 1].set_ylabel("Harrell's C"); a[1, 1].set_ylim(0.5, 0.85); a[1, 1].legend(frameon=False, fontsize=7.5)
for i in range(2):
    a[1, 1].annotate(f"{mm_c[i]:.3f}", (xx[i] - wdt / 2, mm_c[i]), ha="center", va="bottom", fontsize=7)
    a[1, 1].annotate(f"{ee_c[i]:.3f}", (xx[i] + wdt / 2, ee_c[i]), ha="center", va="bottom", fontsize=7)
for ax_ in a.ravel(): ax_.grid(axis="y", alpha=.22, lw=.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_4.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure_4.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig); print("[OK] Figure_4")

# ---------------- Figure S1: BIC（无图内标题、无图内说明、无重叠） ----------------
bic = json.load(open(os.path.join(BASE, "bic_series_binary.json")))["bic"]
ks = sorted(int(k) for k in bic)
vals = [bic[str(k)] for k in ks]
fig, a = plt.subplots(figsize=(4.8, 3.2))
a.plot(ks, vals, "o-", color="#1f4e9c", lw=1.5, ms=4.5)
a.axvline(6, color="#b02418", ls="--", lw=1.0)
for k, v in zip(ks, vals):
    dy = 6000 if k not in (2,) else -11000
    a.annotate(f"{v:,.0f}", (k, v), xytext=(0, dy), textcoords="offset points",
               ha="center", fontsize=6.6, color="#333333")
a.set_xlabel("Number of mixture components (K)")
a.set_ylabel("Bayesian information criterion")
a.set_xticks(ks); a.set_ylim(min(vals) - 26000, max(vals) + 16000)
a.grid(alpha=.25, lw=.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_S1.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure_S1.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig); print("[OK] Figure_S1")

# ---------------- Figure S3: 交叉验证 ----------------
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
rows_a, rows_b = [], []
for rep, (tr_i, te_i) in enumerate(rskf.split(m, m["mort_28d"])):
    dtr, dte = m.iloc[tr_i], m.iloc[te_i]
    sc2 = StandardScaler()
    Xtr = sc2.fit_transform(np.column_stack([dtr[D].ffill(axis=1).bfill(axis=1).values, dtr[V].values]))
    Xte = sc2.transform(np.column_stack([dte[D].ffill(axis=1).bfill(axis=1).values, dte[V].values]))
    g = GaussianMixture(6, covariance_type="full", random_state=SEED, n_init=10, max_iter=300, tol=1e-3).fit(Xtr)
    trc, tec = g.predict(Xtr), g.predict(Xte)
    yt = dtr["mort_28d"].values
    ordr = sorted(range(6), key=lambda c: (yt[trc == c].mean() if (trc == c).sum() else 9))
    mp = {o: n for n, o in enumerate(ordr)}
    trc = np.array([mp[c] for c in trc]); tec = np.array([mp[c] for c in tec])
    d1 = pd.DataFrame({"time": dtr["surv"].values, "event": yt})
    d2 = pd.DataFrame({"time": dte["surv"].values, "event": dte["mort_28d"].values})
    for c in range(1, 6):
        d1[f"c{c}"] = (trc == c).astype(int); d2[f"c{c}"] = (tec == c).astype(int)
    cp1 = CoxPHFitter(penalizer=0.5).fit(d1, "time", "event")
    lp = cp1.predict_partial_hazard(d2.drop(columns=["time", "event"]))
    rows_a.append(concordance_index(d2["time"].values, -lp.values, d2["event"].values))
    d1["sofa"] = dtr["sofa"].values; d2["sofa"] = dte["sofa"].values
    cp2 = CoxPHFitter(penalizer=0.5).fit(d1, "time", "event")
    lp2 = cp2.predict_partial_hazard(d2.drop(columns=["time", "event"]))
    rows_b.append(concordance_index(d2["time"].values, -lp2.values, d2["event"].values))
A = np.array(rows_a).reshape(5, 5); B = np.array(rows_b).reshape(5, 5)
print(f"CV class-only {A.mean():.3f}±{A.std():.3f}; class+SOFA {B.mean():.3f}±{B.std():.3f}")
fig, ax2 = plt.subplots(1, 2, figsize=(7.2, 3.0))
for axx, Mx, ttl, lo, hi in [(ax2[0], A, "Class only", 0.60, 0.82), (ax2[1], B, "Class + SOFA", 0.68, 0.88)]:
    im = axx.imshow(Mx, cmap="Blues", vmin=lo, vmax=hi, aspect="auto")
    axx.set_xticks(range(5)); axx.set_xticklabels([f"Fold {i+1}" for i in range(5)], fontsize=7.5)
    axx.set_yticks(range(5)); axx.set_yticklabels([f"Rep {i+1}" for i in range(5)], fontsize=7.5)
    for i in range(5):
        for j in range(5):
            axx.annotate(f"{Mx[i,j]:.3f}", (j, i), ha="center", va="center", fontsize=6.6,
                         color="#111111" if Mx[i, j] < hi - (hi - lo) * 0.45 else "#ffffff")
    axx.set_title(ttl, fontsize=8.5)
    plt.colorbar(im, ax=axx, fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_S3.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure_S3.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig); print("[OK] Figure_S3")

json.dump({"cv_class_only": [round(float(A.mean()), 4), round(float(A.std()), 4)],
           "cv_class_sofa": [round(float(B.mean()), 4), round(float(B.std()), 4)],
           "fig4_mimic_pct": [round(float(x), 2) for x in mm_m],
           "fig4_eicu_pct": [round(float(x), 2) for x in mm_e],
           "fig4_c": {"mimic": [round(float(x), 4) for x in mm_c], "eicu": [round(float(x), 4) for x in ee_c]},
           "mimic_class_n": {f"C{c}": int((cl == c).sum()) for c in range(6)},
           "eicu_class_n": {f"C{c}": int((ecl == c).sum()) for c in range(6)}},
          open(os.path.join(BASE, "rev_fig_results.json"), "w"), indent=2)
print("[SAVED] rev_fig_results.json")
print("[DONE]")
