# -*- coding: utf-8 -*-
"""方案 A（单位修正版）：eICU 完整重跑。

背景
----
原 eICU 血管活性药剂量把 '(mcg/min)' 分支误写成 `rate / 60 / weight`
（见 recovered/eicu_pipeline/02_eicu_cont_dose_threshold.py:62-63），
相当于把 mcg/min 当成 mcg/h，剂量被系统性压低 60 倍，导致大量真实
输注窗跌到阳性阈值以下（旧口径 38.6% 患者 8 窗全零）。

本脚本
------
1. 从 eicu_crd.infusiondrug 重取 NE 记录，按**正确**单位换算重建
   连续剂量 dose_w（mcg/kg/min 当量）与二值 vaso_w（> 0.01）；
2. DBP 取自源表（vitalperiodic + vitalaperiodic，未填补），缺失用 LOCF；
3. 队列保持原始 3,624；
4. 重跑 GMM K=6 类划分、描述统计、Cox（penalizer=0）与 bootstrap CI、
   48h landmark，以及 DBP 构造成分独立增量（Table 5 的 eICU 列）；
5. 附带 K=5 敏感性。

输出：eicu_planA_v2_results.json / eicu_planA_v2_constructs.json /
      eicu_features_planA_v2.csv
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import pg8000
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from lifelines import CoxPHFitter

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


B = DATA_DIR
D = [f"dbp_w{i}" for i in range(8)]
V = [f"vaso_w{i}" for i in range(8)]
DS = [f"dose_w{i}" for i in range(8)]
K = 6

# ---------- 1. 队列主体（pid / mort / sofa / los_days / lactate） ----------
e = pd.read_csv(os.path.join(B, "recovered", "validate", "eicu_diag_features_cont.csv"))
e["pid"] = e["pid"].astype(int)
e = e[["pid", "age", "los_days", "mort", "visit", "sofa", "lactate", "weight"]].copy()
pids = e["pid"].tolist()
print(f"[队列] 原始 eICU 队列 n = {len(e)}  院内死亡 {e['mort'].mean():.1%}  有 SOFA {e['sofa'].notna().sum()}")

# ---------- 2. 重取 NE 记录并正确换算 ----------
conn = pg8000.connect(host="localhost", port=5433, user="postgres",
                      password=DB_PASSWORD, database="eicu")
cur = conn.cursor()
rows, W = [], {}
for i in range(0, len(pids), 400):
    b = pids[i:i + 400]; bh = ",".join(["%s"] * len(b))
    cur.execute(f"SELECT patientunitstayid, drugname, drugrate, infusionoffset "
                f"FROM eicu_crd.infusiondrug WHERE patientunitstayid IN ({bh}) "
                f"AND drugname ILIKE '%norepinephrine%'", b)
    rows += cur.fetchall()
    cur.execute(f"SELECT patientunitstayid, admissionweight FROM eicu_crd.patient "
                f"WHERE patientunitstayid IN ({bh})", b)
    for r in cur.fetchall():
        try: W[r[0]] = float(r[1])
        except Exception: W[r[0]] = np.nan
conn.close()

d = pd.DataFrame(rows, columns=["pid", "drug", "rate", "offset"])
d["rate_n"] = pd.to_numeric(d.rate, errors="coerce")
d["off_n"] = pd.to_numeric(d.offset, errors="coerce").fillna(0)
d["w"] = np.floor(d.off_n / 360).astype(int)
d = d[(d.w >= 0) & (d.w <= 7)].copy()
d["weight"] = pd.to_numeric(d.pid.map(W), errors="coerce").fillna(70)
d.loc[d.weight <= 0, "weight"] = 70

u = d.drug.str.lower()
d["u"] = np.select(
    [u.str.contains(r"\(mcg/kg/min\)"), u.str.contains(r"\(mcg/min\)"), u.str.contains(r"\(ml/hr\)"),
     u.str.contains(r"\(mg/hr\)"), u.str.contains(r"\(mcg/hr\)")],
    ["mcg/kg/min", "mcg/min", "ml/hr", "mg/hr", "mcg/hr"], default="other")
r = d.rate_n; wt = d.weight
d["dose"] = np.select(
    [d.u.eq("mcg/kg/min"), d.u.eq("mcg/min"), d.u.eq("ml/hr"), d.u.eq("mg/hr"), d.u.eq("mcg/hr")],
    [r, r / wt, r * 32 / 60 / wt, r * 1000 / 60 / wt, r / 60 / wt], default=r * 32 / 60 / wt)
d.loc[r.isna() | (r <= 0), "dose"] = 0.0
d["dose"] = d["dose"].clip(0, 5)

print(f"[剂量] NE 记录 {len(rows):,} 条；单位构成（按记录 / 按覆盖患者）:")
uniq = d.u.value_counts()
for k, v in uniq.items():
    print(f"    {k:12s} 记录 {int(v):6d}   覆盖患者 {d[d.u == k].pid.nunique():5d}")

piv = (d.groupby(["pid", "w"])["dose"].mean().reset_index()
         .pivot_table(index="pid", columns="w", values="dose")
         .reindex(pids, columns=range(8)))
piv.index = e.index
for w in range(8):
    e[f"dose_w{w}"] = piv[w].values
    e[f"vaso_w{w}"] = (piv[w].fillna(0) > 0.01).astype(int)

print("[剂量] 修正后 vaso_w 各窗阳性率:",
      [round(float(e[c].mean()) * 100, 1) for c in V])
print(f"[剂量] 连续剂量透视：非空 {int(piv.notna().sum().sum())}/{piv.size}")
nz = int((e[V].sum(axis=1) == 0).sum())
print(f"[剂量] 修正后 8 窗全零患者: {nz} ({nz/len(e)*100:.1f}%)")

# ---------- 3. 源表 DBP + LOCF ----------
raw = pd.read_pickle(os.path.join(B, "recovered", "eicu_dbp_source_raw.pkl")).copy()
raw["w"] = np.floor(raw["off"] / 360).astype(int)
raw = raw[(raw.w >= 0) & (raw.w <= 7)]
dp = (raw.groupby(["pid", "w"])["dbp"].mean().reset_index()
        .pivot_table(index="pid", columns="w", values="dbp").reindex(pids, columns=range(8)))
dp.index = e.index
e = e.drop(columns=[c for c in D if c in e.columns]).join(
    dp.rename(columns={w: f"dbp_w{w}" for w in range(8)}))
miss_raw = {f"W{w}": round(float(dp[w].isna().mean()) * 100, 1) for w in range(8)}
nwin = dp.notna().sum(axis=1)
print("[DBP] 源表口径各窗缺失率:", miss_raw)
e = e[dp.notna().sum(axis=1).values >= 4].reset_index(drop=True)
print(f"[队列] ≥4 个可评估 DBP 窗后 n = {len(e)}")

e["surv"] = e["los_days"].clip(lower=0.01, upper=28.0)
e["mort"] = e["mort"].astype(int)
L = e[D].ffill(axis=1).bfill(axis=1)

# ---------- 4. GMM ----------
def gmm_cl(X, k, y):
    sc = StandardScaler()
    g = GaussianMixture(k, covariance_type="full", random_state=42, n_init=20,
                        max_iter=500, tol=1e-4).fit(sc.fit_transform(X))
    cl = g.predict(sc.transform(X))
    o = sorted(range(k), key=lambda c: (y[cl == c].mean() if (cl == c).sum() else 9))
    mp = {a: b for b, a in enumerate(o)}
    return np.array([mp[c] for c in cl])

Xin = np.column_stack([L.values, e[V].values])
y = e["mort"].values.astype(int)
e["cls"] = gmm_cl(Xin, K, y)

print(f"\n=== eICU 类别特征（方案 A + 单位修正，K={K}）===")
print(f"{'类':4s} {'n':>6s} {'%':>6s} {'SOFA':>6s} {'乳酸':>6s} {'LOS中位':>8s} {'死亡%':>7s} {'VP窗':>6s}")
rowsx = []
for c in range(K):
    s = e[e.cls == c]
    n = len(s); k = int(s.mort.sum()); p = k / n
    lo = max(0.0, (2*n*p + 1.96**2 - 1.96*np.sqrt(4*n*p*(1-p) + 1.96**2)) / (2*(n + 1.96**2)))
    hi = min(1.0, (2*n*p + 1.96**2 + 1.96*np.sqrt(4*n*p*(1-p) + 1.96**2)) / (2*(n + 1.96**2)))
    rowsx.append({
        "class": f"C{c}", "n": n, "pct": round(n/len(e)*100, 1),
        "sofa": round(float(s.sofa.mean()), 1), "sofa_sd": round(float(s.sofa.std()), 1),
        "lactate": round(float(s.lactate.mean()), 1), "lactate_sd": round(float(s.lactate.std()), 1),
        "los_med": round(float(s.los_days.median()), 1),
        "los_q1": round(float(s.los_days.quantile(.25)), 1),
        "los_q3": round(float(s.los_days.quantile(.75)), 1),
        "mort_n": k, "mort_pct": round(p*100, 1),
        "mort_lo": round(lo*100, 1), "mort_hi": round(hi*100, 1),
        "vp_windows": round(float(s[V].sum(axis=1).mean()), 1),
        "vp_windows_sd": round(float(s[V].sum(axis=1).std()), 1),
        "dbp_w0": round(float(L.loc[e.cls == c, "dbp_w0"].mean()), 1),
        "dbp_w0_sd": round(float(L.loc[e.cls == c, "dbp_w0"].std()), 1),
        "dbp_w4": round(float(L.loc[e.cls == c, "dbp_w4"].mean()), 1),
        "dbp_w4_sd": round(float(L.loc[e.cls == c, "dbp_w4"].std()), 1),
    })
    print(f"{rowsx[-1]['class']:4s} {n:6d} {rowsx[-1]['pct']:6.1f} {rowsx[-1]['sofa']:6.1f} "
          f"{rowsx[-1]['lactate']:6.1f} {rowsx[-1]['los_med']:8.1f} {rowsx[-1]['mort_pct']:7.1f} "
          f"{rowsx[-1]['vp_windows']:6.1f}")
tab = pd.DataFrame(rowsx)

# ---------- 5. Cox + bootstrap CI ----------
def cfit(t, ev, Xd, pen=0.0):
    dd = pd.DataFrame({"time": np.asarray(t, float), "event": np.asarray(ev, int)})
    for c in Xd.columns:
        dd[c] = np.asarray(Xd[c].values, float)
    return CoxPHFitter(penalizer=pen).fit(dd, "time", "event")

ok = e["sofa"].notna().values
s = e[ok].reset_index(drop=True)
cl_s = e.loc[ok, "cls"].values
Xcls = pd.DataFrame({f"c{c}": (cl_s == c).astype(float) for c in range(1, K)})
m_cls = cfit(s.surv, s.mort, Xcls)
m_sofa = cfit(s.surv, s.mort, pd.DataFrame({"sofa": s.sofa.values}))
m_both = cfit(s.surv, s.mort, Xcls.assign(sofa=s.sofa.values))
c_cls, c_sofa, c_both = m_cls.concordance_index_, m_sofa.concordance_index_, m_both.concordance_index_
print(f"\n=== Cox（n={len(s)} 有 SOFA）===")
print(f"  C(class only)={c_cls:.4f}  C(SOFA)={c_sofa:.4f}  C(class+SOFA)={c_both:.4f}  ΔC={c_both-c_sofa:+.4f}")

rng = np.random.default_rng(11)
oc, oo, ob, od = [], [], [], []
n = len(s)
for _ in range(400):
    i = rng.integers(0, n, n)
    try:
        dd = pd.DataFrame({"time": s.surv.values.astype(float)[i], "event": s.mort.values.astype(int)[i],
                           "sofa": s.sofa.values.astype(float)[i]})
        for c in range(1, K):
            dd[f"c{c}"] = (cl_s[i] == c).astype(float)
        a = CoxPHFitter(penalizer=0.0).fit(dd, "time", "event").concordance_index_
        b = CoxPHFitter(penalizer=0.0).fit(dd[["time", "event", "sofa"]], "time", "event").concordance_index_
        z2 = CoxPHFitter(penalizer=0.0).fit(dd[["time", "event"]].assign(
            **{f"c{c}": (cl_s[i] == c).astype(float) for c in range(1, K)}), "time", "event").concordance_index_
        oc.append(z2); ob.append(a); oo.append(b); od.append(a - b)
    except Exception:
        pass
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
print(f"  C(class only) 95%CI={ci(oc)}   C(SOFA) 95%CI={ci(oo)}")
print(f"  C(class+SOFA) 95%CI={ci(ob)}   ΔC 95%CI={ci(od)}  (bootstrap n={len(od)})")

# ---------- 6. 48h landmark ----------
keep = e["surv"].values > 2.0
e3 = e[keep].reset_index(drop=True)
s3 = e3[e3.sofa.notna()].reset_index(drop=True)
Xc3 = pd.DataFrame({f"c{c}": (s3.cls == c).astype(float) for c in range(1, K)})
a3 = cfit(s3.surv - 2.0, s3.mort, Xc3.assign(sofa=s3.sofa.values))
b3 = cfit(s3.surv - 2.0, s3.mort, pd.DataFrame({"sofa": s3.sofa.values}))
print(f"\n=== 48h landmark（排除 {int((~keep).sum())} 例，{100*(~keep).mean():.1f}%）===")
print(f"  n_cohort={int(keep.sum())}  n_sofa={len(s3)}  C(class+SOFA)={a3.concordance_index_:.4f}  "
      f"C(SOFA)={b3.concordance_index_:.4f}  ΔC={a3.concordance_index_-b3.concordance_index_:+.4f}")
rng2 = np.random.default_rng(12)
o2 = []
for _ in range(400):
    i = rng2.integers(0, len(s3), len(s3))
    try:
        dd = pd.DataFrame({"time": (s3.surv - 2.0).values.astype(float)[i],
                           "event": s3.mort.values.astype(int)[i],
                           "sofa": s3.sofa.values.astype(float)[i]})
        for c in range(1, K):
            dd[f"c{c}"] = (s3.cls.values[i] == c).astype(float)
        x = CoxPHFitter(penalizer=0.0).fit(dd, "time", "event").concordance_index_
        z2 = CoxPHFitter(penalizer=0.0).fit(dd[["time", "event", "sofa"]], "time", "event").concordance_index_
        o2.append(x - z2)
    except Exception:
        pass
print(f"  landmark ΔC 95%CI={ci(o2)}")

res = {
    "n_cohort": int(len(e)), "n_sofa_complete": int(len(s)),
    "mort_overall": round(float(e.mort.mean())*100, 1),
    "vaso_window_pct": {f"W{w}": round(float(e[f'vaso_w{w}'].mean())*100, 1) for w in range(8)},
    "all_zero_patients": int(nz), "all_zero_pct": round(nz/len(e)*100, 1),
    "unit_mix": {k: int(v) for k, v in uniq.items()},
    "class_n": [int((e.cls == c).sum()) for c in range(K)],
    "mort_range": [round(float(tab.mort_pct.min()), 1), round(float(tab.mort_pct.max()), 1)],
    "c_class_only": round(c_cls, 4), "c_class_only_ci": ci(oc),
    "c_sofa": round(c_sofa, 4), "c_sofa_ci": ci(oo),
    "c_class_sofa": round(c_both, 4), "c_class_sofa_ci": ci(ob),
    "dC": round(c_both - c_sofa, 4), "dC_ci": ci(od),
    "missing_raw_pct": miss_raw, "mean_evaluable_windows": round(float(nwin.mean()), 2),
    "table": rowsx,
    "landmark": {
        "n_excluded": int((~keep).sum()), "n_excluded_pct": round(float((~keep).mean())*100, 1),
        "n_cohort": int(keep.sum()), "n_sofa_complete": int(len(s3)),
        "c_class_sofa": round(a3.concordance_index_, 4), "c_sofa": round(b3.concordance_index_, 4),
        "dC": round(a3.concordance_index_ - b3.concordance_index_, 4), "dC_ci": ci(o2),
    },
}

# ---------- 7. K=5 敏感性 ----------
e["cls5"] = gmm_cl(Xin, 5, y)
cl5 = e.loc[ok, "cls5"].values
X5 = pd.DataFrame({f"c{c}": (cl5 == c).astype(float) for c in range(1, 5)})
a5 = cfit(s.surv, s.mort, X5.assign(sofa=s.sofa.values)).concordance_index_
b5 = cfit(s.surv, s.mort, X5).concordance_index_
print(f"\n=== K=5 敏感性 ===\n  C(class only)={b5:.4f}  C(class+SOFA)={a5:.4f}  ΔC={a5-b5:+.4f}")
print(f"  各类 n: {[int((e.cls5==c).sum()) for c in range(5)]}  "
      f"死亡%: {[round(float(y[e.cls5.values==c].mean()*100),1) for c in range(5)]}")
res["sensitivity_k5"] = {
    "class_n": [int((e.cls5 == c).sum()) for c in range(5)],
    "mort_pct": [round(float(y[e.cls5.values == c].mean()*100), 1) for c in range(5)],
    "c_class_only": round(b5, 4), "c_class_sofa": round(a5, 4), "dC": round(a5 - b5, 4)}

with open(os.path.join(B, "eicu_planA_v2_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("\n[SAVED] eicu_planA_v2_results.json")

# ---------- 8. DBP 构造成分独立增量（Table 5 eICU 列） ----------
# eICU 有 81.8% 患者八窗全覆盖，可评估窗数近乎退化；按既定口径（补充表 S10
# 脚注），对该队列的参考模型与每个增广模型**统一**施加 ridge 0.01，使 χ² 在
# 相同惩罚下可比。
PEN_EICU = 0.01


def fit_cox(d_, pen=PEN_EICU):
    try:
        return CoxPHFitter(penalizer=pen).fit(d_, "time", "event"), pen
    except Exception as ex:
        print(f"    [fit_cox] pen={pen} 失败: {type(ex).__name__}: {str(ex)[:90]}")
        return None, None

def cox_metrics(t_, e_, X):
    dd = pd.DataFrame({"time": np.asarray(t_, float), "event": np.asarray(e_, int)})
    for c in X.columns: dd[c] = np.asarray(X[c].values, float)
    cp, pen = fit_cox(dd, PEN_EICU)
    if cp is None: return None
    ll = float(cp.log_likelihood_)
    return dict(C=round(float(cp.concordance_index_), 4), loglik=round(ll, 2),
                AIC=round(-2*ll + 2*(dd.shape[1]-2), 2), npar=dd.shape[1]-2, pen=pen)

def dynamics(A):
    nn = A.shape[0]
    mean = np.nanmean(A, 1); sd = np.nanstd(A, 1, ddof=1)
    mn = np.nanmin(A, 1); mx = np.nanmax(A, 1)
    nobs = np.sum(~np.isnan(A), 1)
    b40 = np.nansum(A < 40, 1) / np.maximum(nobs, 1)
    b50 = np.nansum(A < 50, 1) / np.maximum(nobs, 1)
    slope = np.full(nn, np.nan); idx = np.arange(8.0)
    for i in range(nn):
        v = A[i]; okk = ~np.isnan(v)
        if okk.sum() >= 3: slope[i] = np.polyfit(idx[okk], v[okk], 1)[0]
    slope = np.where(np.isnan(slope), np.nanmedian(slope), slope)
    return dict(mean=mean, sd=sd, min=mn, rng=mx-mn, nobs=nobs, b40=b40, b50=b50, slope=slope)

def z(a):
    # ndarray.mean()/std() do not skip NaN; the corrected continuous dose
    # series legitimately contains NaN (windows with no infusion record), so
    # the nan-aware versions are required here.
    a = np.asarray(a, float)
    sd_ = np.nanstd(a)
    return (a - np.nanmean(a)) / (sd_ if sd_ > 0 else 1)

Dobs = e.loc[ok, D].values.astype(float)
t = s.surv.values.astype(float); yv = s.mort.values.astype(int)
sofa = s.sofa.values.astype(float)
dose_mean = s[DS].mean(axis=1).values
# Patients with no infusion record in any window are coded as zero exposure
dose_mean = np.where(np.isnan(dose_mean), 0.0, dose_mean)
print(f"[参考模型] dose 均值：0 值 {int((dose_mean == 0).sum())}/{len(dose_mean)} 例，"
      f"中位 {np.median(dose_mean):.4f}，最大 {dose_mean.max():.3f}")
# 参考模型中的「血管药类别」= 仅用 8 个血管药窗独立聚出的类（对应 S9 的
# vasopressor-only 区块），与主分析的联合 DBP+血管药类不同。
cl_v = gmm_cl(s[V].values, K, yv)
print(f"[参考模型] 血管药-only 类别 n: {[int((cl_v == c).sum()) for c in range(K)]}")

base = pd.DataFrame({"sofa": sofa})
base = pd.concat([base, pd.DataFrame({f"v{c}": (cl_v == c).astype(float) for c in range(1, K)})], axis=1)
base["dosez"] = z(dose_mean); base["dosez2"] = z(dose_mean)**2
dy = dynamics(Dobs)
base["nobs"] = z(dy["nobs"])
bm = cox_metrics(t, yv, base)
print(f"\n=== Table 5（eICU 列，单位修正后）===")
print(f"基准 SOFA+类别+剂量+剂量²+窗数 → C={bm['C']:.4f} AIC={bm['AIC']:.1f}")

cons = {"Average level": dy["mean"], "Minimum": dy["min"], "Standard deviation": dy["sd"],
        "48-hour slope": dy["slope"], "Range": dy["rng"],
        "Proportion below 50 mmHg": dy["b50"], "Proportion below 40 mmHg": dy["b40"]}
recs = {}
for name, v in cons.items():
    X = base.copy(); X["x"] = np.asarray(v, float)
    mm = cox_metrics(t, yv, X)
    recs[name] = {"C": mm["C"], "dC": round(mm["C"] - bm["C"], 4),
                  "dAIC": round(mm["AIC"] - bm["AIC"], 1),
                  "chi2": round(2*(mm["loglik"] - bm["loglik"]), 1)}
    print(f"  {name:28s} C={mm['C']:.4f}  ΔC={recs[name]['dC']:+.4f}  ΔAIC={recs[name]['dAIC']:+.1f}  "
          f"χ²={recs[name]['chi2']:.1f}")
comp = base.copy()
for k, key in [("dbp_sd", "sd"), ("dbp_slope", "slope"), ("dbp_b40", "b40"),
               ("dbp_mean", "mean"), ("dbp_min", "min")]:
    comp[k] = z(dy[key])
mc = cox_metrics(t, yv, comp)
print(f"  五变量动态区块              C={mc['C']:.4f}  ΔC={mc['C']-bm['C']:+.4f}  "
      f"χ²={2*(mc['loglik']-bm['loglik']):.1f}")
Xi = comp.copy()
for i in range(K-1):
    Xi[f"vd{i}"] = comp[f"v{i+1}"].values * z(dy["sd"])
mi = cox_metrics(t, yv, Xi)
print(f"  交互（类别×DBP不稳定）      χ²={2*(mi['loglik']-mc['loglik']):.1f}  C={mi['C']:.4f}")

out = {"n": len(s), "base": bm, "constructs": recs,
       "block_dynamics": {"C": mc["C"], "dC": round(mc["C"]-bm["C"], 4),
                          "dAIC": round(mc["AIC"]-bm["AIC"], 1),
                          "chi2": round(2*(mc["loglik"]-bm["loglik"]), 1)},
       "interaction": {"C": mi["C"], "chi2": round(2*(mi["loglik"]-mc["loglik"]), 1)}}
with open(os.path.join(B, "eicu_planA_v2_constructs.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n[SAVED] eicu_planA_v2_constructs.json")

e.to_csv(os.path.join(B, "eicu_features_planA_v2.csv"), index=False)
print("[SAVED] eicu_features_planA_v2.csv")
