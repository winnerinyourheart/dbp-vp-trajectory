# -*- coding: utf-8 -*-
"""eICU 口径统一 + Figure 1 / Figure 4 重制（图例移出绘图区）"""
import sys, os, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from lifelines import CoxPHFitter

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


BASE = DATA_DIR
OUT = os.path.join(BASE, "submission_clep", "Figures_rev")
D = [f"dbp_w{i}" for i in range(8)]; V = [f"vaso_w{i}" for i in range(8)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8})
CMAP = ["#1f4e9c", "#3d7ebf", "#7fb3d9", "#f2b04a", "#e0732a", "#b02418"]
LBL = [f"C{i}" for i in range(6)]

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round((c - h) * 100, 1), round((c + h) * 100, 1)

def fit_order(X, K, y, seed=42, n_init=20):
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    g = GaussianMixture(K, covariance_type="full", random_state=seed, n_init=n_init,
                        max_iter=500, tol=1e-4).fit(Xs)
    cl = g.predict(Xs)
    o = sorted(range(K), key=lambda c: (y[cl == c].mean() if (cl == c).sum() else 9))
    mp = {a: b for b, a in enumerate(o)}
    return np.array([mp[c] for c in cl])

def cstat(t, e, Xd, pen=0.0):
    d = pd.DataFrame({"time": np.asarray(t, float), "event": np.asarray(e, int)})
    for c in Xd.columns: d[c] = Xd[c].values
    for p in [pen, 0.01, 0.1, 0.5]:
        if p < pen: continue
        try:
            return CoxPHFitter(penalizer=p).fit(d, "time", "event").concordance_index_
        except Exception: continue
    raise RuntimeError("Cox 拟合失败")

def des(cl, K): return pd.DataFrame({f"c{c}": (cl == c).astype(float) for c in range(1, K)})

# ---------- MIMIC ----------
m = pd.read_csv(os.path.join(BASE, "binary_mimic_data.csv")); m["surv"] = m["surv"].clip(lower=0.01)
Db = m[D].ffill(axis=1).bfill(axis=1)
Xm = np.column_stack([Db.values, m[V].values])
y = m["mort_28d"].values.astype(int)
clm = fit_order(Xm, 6, y); m["cls"] = clm
sm = m["sofa"].values.astype(float); svm = m["surv"].values.astype(float)

# ---------- eICU：全队列（n=3539）用于分类与死亡率 ----------
ee = pickle.load(open(os.path.join(BASE, "eicu_gmm_binary.pkl"), "rb"))["data"].copy()
raw = pd.read_csv(os.path.join(BASE, "eicu_dbp_windows_raw.csv"))
ee = ee.drop(columns=D).merge(raw, on="pid", how="left")
ee = ee[ee[D].notna().sum(axis=1) >= 4].reset_index(drop=True)
ee["surv"] = ee["los_days"].clip(lower=0.01, upper=28.0)
ee["mort"] = ee["mort"].astype(int)
Xe = np.column_stack([ee[D].ffill(axis=1).bfill(axis=1).values, ee[V].values])
ye = ee["mort"].values.astype(int)
cle = fit_order(Xe, 6, ye); ee["cls"] = cle
print(f"eICU 全队列 n={len(ee)}")

res = {"eicu_n_full": int(len(ee)), "mimic_n": int(len(m))}
mort_e = []
print("\neICU 全队列各类（死亡率 + Wilson 95% CI）:")
for c in range(6):
    k = int(ye[cle == c].sum()); n = int((cle == c).sum())
    lo, hi = wilson(k, n)
    mort_e.append({"class": f"C{c}", "n": n, "events": k, "pct": round(k / n * 100, 1), "ci": [lo, hi],
                   "sofa": round(float(ee.sofa[cle == c].mean()), 2)})
    print(f"  C{c}: {k}/{n} = {k/n*100:.1f}% (95%CI {lo}-{hi})  SOFA {ee.sofa[cle==c].mean():.2f}")
res["eicu_class_mort_full"] = mort_e
mort_m = []
for c in range(6):
    k = int(y[clm == c].sum()); n = int((clm == c).sum())
    lo, hi = wilson(k, n)
    mort_m.append({"class": f"C{c}", "n": n, "events": k, "pct": round(k / n * 100, 1), "ci": [lo, hi]})
res["mimic_class_mort"] = mort_m

# eICU SOFA 完整子集 Cox
ee2 = ee[ee["sofa"].notna()].reset_index(drop=True)
Xc2 = des(ee2["cls"].values, 6)
sv2 = ee2["surv"].values.astype(float); sf2 = ee2["sofa"].values.astype(float); y2 = ee2["mort"].values.astype(int)
cU = cstat(sv2, y2, Xc2)
c0 = cstat(sv2, y2, pd.DataFrame(index=range(len(ee2))).assign(sofa=sf2))
c1 = cstat(sv2, y2, Xc2.assign(sofa=sf2))
print(f"\neICU SOFA 完整子集 n={len(ee2)}: C(class)={cU:.4f}  C(SOFA)={c0:.4f}  C(class+SOFA)={c1:.4f}  dC={c1-c0:+.4f}")
res["eicu_sofa_subset"] = {"n": int(len(ee2)), "c_class_only": round(cU, 4), "c_sofa": round(c0, 4),
                           "c_class_sofa": round(c1, 4), "dC": round(c1 - c0, 4)}

# ---------- Figure 1（共享图例，移出绘图区） ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.35))
for _i, _l in enumerate(["A", "B"]):
    ax[_i].set_title(_l, loc="left", fontweight="bold", fontsize=10)
w = np.arange(8)
for c in range(6):
    d = m[m.cls == c]
    ax[0].plot(w, d[D].mean().values, color=CMAP[c], marker="o", ms=3.0, lw=1.4)
    ax[1].plot(w, d[V].mean().values * 100, color=CMAP[c], marker="o", ms=3.0, lw=1.4)
ax[0].set_ylabel("Mean DBP (mmHg)"); ax[1].set_ylabel("Patients on vasopressors (%)")
for a in ax:
    a.set_xlabel("6-hour window (0\u201348 h from ICU admission)")
    a.set_xticks(w); a.set_xticklabels([f"W{i}" for i in range(8)]); a.grid(alpha=.25, lw=.6)
handles = [plt.Line2D([], [], color=CMAP[c], marker="o", ms=3.2, lw=1.4,
                      label=f"{LBL[c]} (n={int((clm==c).sum())})") for c in range(6)]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.035))
fig.tight_layout(rect=(0, 0.075, 1, 1))
fig.savefig(os.path.join(OUT, "Figure_1.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure_1.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig); print("[OK] Figure_1")

# ---------- Figure 4（panel D 用 SOFA 完整子集） ----------
fig, a = plt.subplots(2, 2, figsize=(7.2, 5.8))
mm_m = [x["pct"] for x in mort_m]; mm_e = [x["pct"] for x in mort_e]
for i, (vals, ttl) in enumerate([(mm_m, "A"), (mm_e, "B")]):
    axx = a[0, i]
    axx.bar(range(6), vals, color=CMAP)
    axx.set_title(ttl, fontsize=10, loc="left", fontweight="bold")
    axx.set_xticks(range(6)); axx.set_xticklabels(LBL); axx.set_ylabel("Mortality (%)")
    for j, v in enumerate(vals):
        axx.annotate(f"{v:.1f}", (j, v), ha="center", va="bottom", fontsize=7)
sm2 = [m.sofa[clm == c].mean() for c in range(6)]
se2 = [ee.sofa[cle == c].mean() for c in range(6)]
a[1, 0].plot(range(6), sm2, "o-", color="#1f4e9c", lw=1.4, ms=4, label="MIMIC-IV")
a[1, 0].plot(range(6), se2, "s--", color="#b02418", lw=1.4, ms=4, label="eICU-CRD")
a[1, 0].set_title("C", fontsize=10, loc="left", fontweight="bold")
a[1, 0].set_xticks(range(6)); a[1, 0].set_xticklabels(LBL); a[1, 0].set_ylabel("Mean SOFA")
a[1, 0].legend(frameon=False, fontsize=7.5)
csm = [cstat(svm, y, des(clm, 6)), cstat(svm, y, des(clm, 6).assign(sofa=sm))]
cse = [cU, c1]
xx = np.arange(2); wdt = 0.34
a[1, 1].bar(xx - wdt / 2, csm, wdt, color="#1f4e9c", label="MIMIC-IV")
a[1, 1].bar(xx + wdt / 2, cse, wdt, color="#b02418", label="eICU-CRD")
a[1, 1].set_xticks(xx); a[1, 1].set_xticklabels(["Class\nonly", "Class\n+ SOFA"])
a[1, 1].set_title("D", fontsize=10, loc="left", fontweight="bold")
a[1, 1].set_ylabel("Harrell's C"); a[1, 1].set_ylim(0.5, 0.85); a[1, 1].legend(frameon=False, fontsize=7.5)
for i in range(2):
    a[1, 1].annotate(f"{csm[i]:.3f}", (xx[i] - wdt / 2, csm[i]), ha="center", va="bottom", fontsize=7)
    a[1, 1].annotate(f"{cse[i]:.3f}", (xx[i] + wdt / 2, cse[i]), ha="center", va="bottom", fontsize=7)
for ax_ in a.ravel(): ax_.grid(axis="y", alpha=.22, lw=.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_4.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure_4.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig); print("[OK] Figure_4")
print("MIMIC C:", [round(x, 4) for x in csm], " eICU C:", [round(x, 4) for x in cse])

json.dump(res, open(os.path.join(BASE, "rev_eicu_final.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("[SAVED] rev_eicu_final.json")
