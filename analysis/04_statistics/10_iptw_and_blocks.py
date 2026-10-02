# -*- coding: utf-8 -*-
"""
CLEP 返修：补充两项分析
 [A] DBP-only 轨迹类别（特征空间剔除血管活性药区块）—— 检验与治疗回路性
 [B] IPTW 倾向评分：协变量平衡(SMD)、权重分布、有效样本量、加权 Cox
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from lifelines import CoxPHFitter
from sqlalchemy import create_engine, text

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


BASE = DATA_DIR
D = [f"dbp_w{i}" for i in range(8)]
V = [f"vaso_w{i}" for i in range(8)]
SEED = 42
M = create_engine(f"postgresql://postgres:{DB_PASSWORD}@localhost:5433/mimiciv")
qm = lambda s: pd.read_sql(text(s), M)
R = {}

def fit_order(X, K, y, seed=SEED, n_init=20):
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    gm = GaussianMixture(K, covariance_type="full", random_state=seed, n_init=n_init,
                         max_iter=500, tol=1e-4).fit(Xs)
    cl = gm.predict(Xs)
    order = sorted(range(K), key=lambda c: (y[cl == c].mean() if (cl == c).sum() else 9))
    mp = {o: n for n, o in enumerate(order)}
    return np.array([mp[c] for c in cl])

def cstat(time, ev, Xdf, pen=0.0):
    d = pd.DataFrame({"time": np.asarray(time, float), "event": np.asarray(ev, int)})
    for c in Xdf.columns: d[c] = Xdf[c].values
    for p in [pen, 0.01, 0.1, 0.5]:
        if p < pen: continue
        try:
            return CoxPHFitter(penalizer=p).fit(d, "time", "event").concordance_index_
        except Exception: continue
    raise RuntimeError("Cox 拟合失败")

m = pd.read_csv(os.path.join(BASE, "binary_mimic_data.csv"))
m["surv"] = m["surv"].clip(lower=0.01)
Db = m[D].ffill(axis=1).bfill(axis=1)
y = m["mort_28d"].values.astype(int)
sofa = m["sofa"].values.astype(float)
surv = m["surv"].values.astype(float)

# 全文主分析（用于 ARI 参照）
cl_full = fit_order(np.column_stack([Db.values, m[V].values]), 6, y)

print("#" * 70); print("# [A] DBP-only 轨迹类别"); print("#" * 70)
cl_dbp = fit_order(Db.values, 6, y)
t = pd.DataFrame({"class": range(6)})
t["n"] = [int((cl_dbp == c).sum()) for c in range(6)]
t["mort"] = [round(float(y[cl_dbp == c].mean()), 4) for c in range(6)]
t["sofa"] = [round(float(sofa[cl_dbp == c].mean()), 2) for c in range(6)]
print(t.to_string(index=False))
Xc = pd.DataFrame({f"c{c}": (cl_dbp == c).astype(float) for c in range(1, 6)})
c_cls = cstat(surv, y, Xc)
c_sofa = cstat(surv, y, pd.DataFrame(index=range(len(y))).assign(sofa=sofa))
c_both = cstat(surv, y, Xc.assign(sofa=sofa))
ari = adjusted_rand_score(cl_full, cl_dbp)
print(f"\nC(class only)={c_cls:.4f}  C(SOFA)={c_sofa:.4f}  C(class+SOFA)={c_both:.4f}  dC={c_both-c_sofa:+.4f}")
print(f"死亡率范围 {min(t.mort):.1%}-{max(t.mort):.1%}   ARI(vs 含血管药区块)={ari:.3f}")
R["dbp_only"] = {"class_n": t["n"].tolist(), "class_mort": t["mort"].tolist(),
                 "class_sofa": t["sofa"].tolist(), "c_class_only": round(c_cls, 4),
                 "c_sofa": round(c_sofa, 4), "c_class_sofa": round(c_both, 4),
                 "dC": round(c_both - c_sofa, 4), "ari_vs_full": round(float(ari), 3),
                 "mort_min": round(float(min(t.mort)), 4), "mort_max": round(float(max(t.mort)), 4)}

print("\n" + "#" * 70); print("# [B] IPTW 倾向评分与协变量平衡"); print("#" * 70)
age = qm("SELECT stay_id, age, gender FROM public.ds_mimic_shock")
ag = qm("""
SELECT ie.stay_id,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%norepinephrine%' THEN 1 ELSE 0 END) ne,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%vasopressin%' THEN 1 ELSE 0 END) vp,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%epinephrine%' AND LOWER(dl.label) NOT LIKE '%norepinephrine%'
           AND LOWER(dl.label) NOT LIKE '%phenylephrine%' THEN 1 ELSE 0 END) epi,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%phenylephrine%' THEN 1 ELSE 0 END) pe,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%dopamine%' THEN 1 ELSE 0 END) da,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%dobutamine%' THEN 1 ELSE 0 END) dob
FROM mimiciv_icu.inputevents ie JOIN mimiciv_icu.d_items dl ON ie.itemid = dl.itemid
WHERE ie.stay_id IN (SELECT DISTINCT stay_id FROM public.ds_mimic_shock)
GROUP BY ie.stay_id""")
d = m[["stay_id"]].merge(age, on="stay_id", how="left").merge(ag, on="stay_id", how="left")
for c in ["ne", "vp", "epi", "pe", "da", "dob"]:
    d[c] = d[c].fillna(0).astype(int)
d["male"] = (d["gender"].astype(str).str.upper().str[0] == "M").astype(int)
d["age"] = d["age"].astype(float)
d["sofa"] = sofa
d["cls"] = cl_full
d["surv"] = surv; d["event"] = y

cov = ["sofa", "age", "male", "ne", "epi", "pe", "da", "dob"]
print(f"vasopressin ever-use: {d.vp.sum()} / {len(d)} ({d.vp.mean():.1%})")
ps = LogisticRegression(max_iter=3000, C=1.0).fit(d[cov].values, d["vp"].values)
p = ps.predict_proba(d[cov].values)[:, 1]
p = np.clip(p, 1e-6, 1 - 1e-6)
pt = d["vp"].mean()
w = np.where(d["vp"] == 1, pt / p, (1 - pt) / (1 - p))
n_trunc = int((w > 10).sum()); w = np.clip(w, None, 10)
ess = (w.sum() ** 2) / (w ** 2).sum()
print(f"稳定化权重: 均值 {w.mean():.2f}, SD {w.std():.2f}, 范围 {w.min():.2f}-{w.max():.2f}, "
      f"截断 {n_trunc} ({n_trunc/len(w)*100:.1f}%), Kish ESS {ess:.0f} ({ess/len(w)*100:.0f}%)")

def smd(x, g, wt=None):
    if wt is None:
        a, b = x[g == 1], x[g == 0]
        ma, mb = a.mean(), b.mean(); va, vb = a.var(ddof=1), b.var(ddof=1)
    else:
        w1, w0 = wt[g == 1], wt[g == 0]
        xa, xb = x[g == 1], x[g == 0]
        ma = np.average(xa, weights=w1); mb = np.average(xb, weights=w0)
        va = np.average((xa - ma) ** 2, weights=w1); vb = np.average((xb - mb) ** 2, weights=w0)
    return float((ma - mb) / np.sqrt((va + vb) / 2 + 1e-12))

bal = {}
print("\n协变量          加权前SMD   加权后SMD")
for c in cov:
    s0, s1 = smd(d[c].values.astype(float), d["vp"].values), smd(d[c].values.astype(float), d["vp"].values, w)
    bal[c] = [round(s0, 3), round(s1, 3)]
    print(f"  {c:<12} {s0:>8.3f} {s1:>11.3f}")
print(f"\n加权前 |SMD| 最大 {max(abs(v[0]) for v in bal.values()):.3f}; "
      f"加权后 |SMD| 最大 {max(abs(v[1]) for v in bal.values()):.3f}")

dd = d[["surv", "event", "sofa", "cls", "vp"]].copy()
for c in range(1, 6):
    dd[f"c{c}"] = (dd["cls"] == c).astype(float)
wcol = ["c1", "c2", "c3", "c4", "c5", "sofa"]
cp = CoxPHFitter().fit(dd[["surv", "event"] + wcol], "surv", "event")
dd["_w"] = w
cpw = CoxPHFitter().fit(dd[["surv", "event", "_w"] + wcol], "surv", "event",
                        weights_col="_w", robust=True)
print("\n未加权 HR (vs C0)        加权 HR (vs C0)")
hr = {}
for c in range(1, 6):
    r0 = cp.summary.loc[f"c{c}"]; r1 = cpw.summary.loc[f"c{c}"]
    h0 = np.exp(r0["coef"]); l0 = np.exp(r0["coef"] - 1.96 * r0["se(coef)"]); u0 = np.exp(r0["coef"] + 1.96 * r0["se(coef)"])
    h1 = np.exp(r1["coef"]); l1 = np.exp(r1["coef"] - 1.96 * r1["se(coef)"]); u1 = np.exp(r1["coef"] + 1.96 * r1["se(coef)"])
    hr[f"C{c}"] = {"unweighted": [round(float(h0), 2), round(float(l0), 2), round(float(u0), 2)],
                   "iptw": [round(float(h1), 2), round(float(l1), 2), round(float(u1), 2)]}
    print(f"  C{c}: {h0:.2f} ({l0:.2f}-{u0:.2f})   |   {h1:.2f} ({l1:.2f}-{u1:.2f})")
r0 = cp.summary.loc["sofa"]; r1 = cpw.summary.loc["sofa"]
print(f"  SOFA/point: {np.exp(r0['coef']):.2f} | {np.exp(r1['coef']):.2f}")
R["iptw"] = {"n_vasopressin": int(d.vp.sum()), "n_total": int(len(d)),
             "p_truncated_pct": round(n_trunc / len(w) * 100, 2),
             "weight_mean": round(float(w.mean()), 2), "weight_sd": round(float(w.std()), 2),
             "weight_range": [round(float(w.min()), 2), round(float(w.max()), 2)],
             "kish_ess": int(round(ess)), "kish_ess_pct": round(ess / len(w) * 100, 1),
             "balance_smd": bal,
             "max_abs_smd_before": round(max(abs(v[0]) for v in bal.values()), 3),
             "max_abs_smd_after": round(max(abs(v[1]) for v in bal.values()), 3),
             "hr": hr}
# SOFA 组间差异
print(f"\nSOFA: 用药组未加权 {d.sofa[d.vp==1].mean():.2f} vs 未用药 {d.sofa[d.vp==0].mean():.2f}; "
      f"加权后 {np.average(d.sofa[d.vp==1],weights=w[d.vp==1]):.2f} vs {np.average(d.sofa[d.vp==0],weights=w[d.vp==0]):.2f}")
R["iptw"]["sofa_unweighted"] = [round(float(d.sofa[d.vp == 1].mean()), 2), round(float(d.sofa[d.vp == 0].mean()), 2)]
R["iptw"]["sofa_weighted"] = [round(float(np.average(d.sofa[d.vp == 1], weights=w[d.vp == 1])), 2),
                              round(float(np.average(d.sofa[d.vp == 0], weights=w[d.vp == 0])), 2)]

json.dump(R, open(os.path.join(BASE, "rev_extra_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n[SAVED] rev_extra_results.json")
print("[DONE]")
