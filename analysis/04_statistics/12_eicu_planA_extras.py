# -*- coding: utf-8 -*-
"""Plan A（单位修正）补充产出：landmark C 的置信区间、S3 表 eICU 的 HR、
S10 的 dose × DBP 交叉分类表。全部基于 eicu_features_planA_v2.csv。"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from lifelines import CoxPHFitter

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


B = DATA_DIR
D = [f"dbp_w{i}" for i in range(8)]
V = [f"vaso_w{i}" for i in range(8)]
DS = [f"dose_w{i}" for i in range(8)]
e = pd.read_csv(os.path.join(B, "eicu_features_planA_v2.csv"))
e["mort"] = e["mort"].astype(int)

ok = e["sofa"].notna().values
s = e[ok].reset_index(drop=True)
cl = s["cls"].values
t = s.surv.values.astype(float)
yv = s.mort.values.astype(int)

out = {}


def cox(t_, ev, X, pen=0.0):
    dd = pd.DataFrame({"time": np.asarray(t_, float), "event": np.asarray(ev, int)})
    for c in X.columns:
        dd[c] = np.asarray(X[c].values, float)
    return CoxPHFitter(penalizer=pen).fit(dd, "time", "event")


# ---------- 1) landmark：C(class+SOFA) 及其 bootstrap CI ----------
keep = e["surv"].values > 2.0
e3 = e[keep].reset_index(drop=True)
s3 = e3[e3.sofa.notna()].reset_index(drop=True)
cl3 = s3.cls.values
Xc3 = pd.DataFrame({f"c{c}": (cl3 == c).astype(float) for c in range(1, 6)})
a3 = cox(s3.surv - 2.0, s3.mort, Xc3.assign(sofa=s3.sofa.values)).concordance_index_
b3 = cox(s3.surv - 2.0, s3.mort, pd.DataFrame({"sofa": s3.sofa.values})).concordance_index_
rng = np.random.default_rng(12)
o3, o3s = [], []
for _ in range(400):
    i = rng.integers(0, len(s3), len(s3))
    try:
        dd = pd.DataFrame({"time": (s3.surv - 2.0).values.astype(float)[i],
                           "event": s3.mort.values.astype(int)[i],
                           "sofa": s3.sofa.values.astype(float)[i]})
        for c in range(1, 6):
            dd[f"c{c}"] = (cl3[i] == c).astype(float)
        o3.append(CoxPHFitter(penalizer=0.0).fit(dd, "time", "event").concordance_index_)
        o3s.append(CoxPHFitter(penalizer=0.0).fit(dd[["time", "event", "sofa"]], "time", "event").concordance_index_)
    except Exception:
        pass
ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
print(f"[landmark] C(class+SOFA)={a3:.4f} 95%CI={ci(o3)}  C(SOFA)={b3:.4f} 95%CI={ci(o3s)}")
out["landmark"] = {"c_class_sofa": round(a3, 4), "c_class_sofa_ci": ci(o3),
                   "c_sofa": round(b3, 4), "c_sofa_ci": ci(o3s), "n": int(len(s3))}

# ---------- 2) S3 表 eICU 的 HR（未惩罚，与脚注一致） ----------
print("\n[S3] eICU 未惩罚 Cox：")
hrs = {}
for name, X in [("A", pd.DataFrame({f"c{c}": (cl == c).astype(float) for c in range(1, 6)})),
                ("B", pd.DataFrame({f"c{c}": (cl == c).astype(float) for c in range(1, 6)}).assign(sofa=s.sofa.values))]:
    m = cox(t, yv, X)
    print(f"  Model {name}: C={m.concordance_index_:.4f}")
    hrs[name] = {"C": round(float(m.concordance_index_), 4), "terms": {}}
    for k, row in m.summary.iterrows():
        lo = np.exp(row["coef"] - 1.96 * row["se(coef)"])
        hi = np.exp(row["coef"] + 1.96 * row["se(coef)"])
        hrs[name]["terms"][k] = [round(float(np.exp(row["coef"])), 2),
                                 round(float(lo), 2), round(float(hi), 2), float(row["p"])]
        print(f"    {k:12s} HR={np.exp(row['coef']):6.2f} ({lo:5.2f}-{hi:5.2f})  p={row['p']:.3g}")
out["hr"] = hrs

# ---------- 3) S10 交叉分类：平均剂量三分位 × 平均 DBP 三分位 ----------
L = e[D].ffill(axis=1).bfill(axis=1)
dbpm = L.mean(axis=1).values
dm = e[DS].mean(axis=1).values
dm = np.where(np.isnan(dm), 0.0, dm)
qd = np.quantile(dm, [1/3, 2/3])
qb = np.quantile(dbpm, [1/3, 2/3])
print(f"\n[S10] 剂量三分位界值 {np.round(qd,4).tolist()}；DBP 三分位界值 {np.round(qb,2).tolist()}")
grp = np.column_stack([
    np.digitize(dm, qd), np.digitize(dbpm, qb)])
tab = np.full((3, 3), np.nan)
for a in range(3):
    for b in range(3):
        m_ = (grp[:, 0] == a) & (grp[:, 1] == b)
        tab[a, b] = e.mort.values[m_].mean() * 100
print("  eICU 剂量(行) × DBP(列) 死亡%：")
print("  " + str(np.round(tab, 1).tolist()))
out["crosstab"] = np.round(tab, 1).tolist()
out["crosstab_n"] = [[int(((grp[:, 0] == a) & (grp[:, 1] == b)).sum()) for b in range(3)] for a in range(3)]

json.dump(out, open(os.path.join(B, "eicu_planA_v2_extras.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n[SAVED] eicu_planA_v2_extras.json")
