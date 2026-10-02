# -*- coding: utf-8 -*-
"""
严谨检验：控制治疗强度后，DBP 的哪些信息是独立于血管活性药的？
- 基线同时控制 SOFA + 血管药类别 + 连续剂量(非线性) + 可评估窗数
- DBP 动态特征用「仅观测窗」计算，避免 LOCF 造成的人为平坦
- MIMIC 与 eICU 双队列
"""
import os, sys, json, warnings, pickle
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
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


def fit_cox(d, pen=0.0):
    for p in [pen, 0.01, 0.1, 0.5, 1.0]:
        if p < pen - 1e-12:
            continue
        try:
            return CoxPHFitter(penalizer=p).fit(d, "time", "event"), p
        except Exception:
            continue
    return None, None


def cox_metrics(t, e, X):
    d = pd.DataFrame({"time": np.asarray(t, float), "event": np.asarray(e, int)})
    for c in X.columns:
        d[c] = np.asarray(X[c].values, float)
    cp, pen = fit_cox(d, 0.0)
    if cp is None:
        return None
    ll = float(cp.log_likelihood_)
    return dict(C=round(float(cp.concordance_index_), 4), loglik=round(ll, 2),
                AIC=round(-2 * ll + 2 * (d.shape[1] - 2), 2), npar=d.shape[1] - 2, pen=pen)


def lrt(a, b):
    if a is None or b is None or a["pen"] != 0 or b["pen"] != 0:
        return "NA"
    return f"{2*(b['loglik']-a['loglik']):.1f}(df{b['npar']-a['npar']})"


def dynamics(A):
    """A: n×8 矩阵，缺失用 nan。仅用观测窗计算动态指标。"""
    n = A.shape[0]
    mean = np.nanmean(A, axis=1)
    sd = np.nanstd(A, axis=1, ddof=1)
    mn = np.nanmin(A, axis=1)
    mx = np.nanmax(A, axis=1)
    nobs = np.sum(~np.isnan(A), axis=1)
    b40 = np.nansum(A < 40, axis=1) / np.maximum(nobs, 1)
    b50 = np.nansum(A < 50, axis=1) / np.maximum(nobs, 1)
    slope = np.full(n, np.nan)
    idx = np.arange(8.0)
    for i in range(n):
        v = A[i]
        ok = ~np.isnan(v)
        if ok.sum() >= 3:
            slope[i] = np.polyfit(idx[ok], v[ok], 1)[0]
    slope = np.where(np.isnan(slope), np.nanmedian(slope), slope)
    return dict(mean=mean, sd=sd, min=mn, rng=mx - mn, nobs=nobs, b40=b40, b50=b50, slope=slope)


def gmm_cl(X, k, y):
    sc = StandardScaler()
    g = GaussianMixture(k, covariance_type="full", random_state=42, n_init=20,
                        max_iter=500, tol=1e-4).fit(sc.fit_transform(X))
    cl = g.predict(sc.transform(X))
    o = sorted(range(k), key=lambda c: (y[cl == c].mean() if (cl == c).sum() else 9))
    mp = {a: b for b, a in enumerate(o)}
    return np.array([mp[c] for c in cl])


def z(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / (s if s > 0 else 1)


def dummies(cl, k, pre=""):
    return pd.DataFrame({f"{pre}g{c}": (cl == c).astype(float) for c in range(1, k)})


def clinic(t, e, sofa, Y, Dobs, dose_mean, cl_vaso, cl_joint, tag, out):
    """核心：在控制治疗强度后，逐个检验 DBP 构造的独立增量"""
    print("\n" + "=" * 84)
    print(f"【{tag}】控制 SOFA + 血管药类别 + 连续剂量(非线性) + 观测窗数后，DBP 各构造的独立增量")
    print("=" * 84)
    dy = dynamics(Dobs)
    dv = dose_mean
    base = pd.DataFrame({"sofa": sofa})
    base = pd.concat([base, dummies(cl_vaso, K, "v")], axis=1)
    base["dosez"] = z(dv)
    base["dosez2"] = z(dv) ** 2
    base["nobs"] = z(dy["nobs"])
    bm = cox_metrics(t, e, base)
    print(f"基准: SOFA + 血管药类别(5) + 剂量 + 剂量² + 观测窗数  →  C={bm['C']:.4f}  AIC={bm['AIC']:.1f}")
    print(f"\n{'DBP 构造（逐个加入）':28s} {'C':>7s} {'ΔC':>8s} {'ΔAIC':>9s} {'LRT χ²(df1)':>13s} {'方向':>6s}")
    cons = {
        "DBP 均值(水平)": dy["mean"],
        "DBP 最低值": dy["min"],
        "DBP 标准差(不稳定)": dy["sd"],
        "DBP 48h 斜率(恢复)": dy["slope"],
        "DBP 极差": dy["rng"],
        "DBP<50 负荷": dy["b50"],
        "DBP<40 负荷": dy["b40"],
        "log(DBP)-log(剂量)": np.log(dy["mean"] + 1) - np.log(dv + 0.02),
        "DBP 剂量残差(线性)": dy["mean"] - np.polyval(np.polyfit(dv, dy["mean"], 1), dv),
        "DBP 秩-剂量秩 差": z(pd.Series(dy["mean"]).rank()) - z(pd.Series(dv).rank()),
    }
    recs = []
    for name, v in cons.items():
        X = base.copy()
        X["x"] = np.asarray(v, float)
        mm = cox_metrics(t, e, X)
        cc = mm["C"] if mm else float("nan")
        dc = round(mm["C"] - bm["C"], 4) if mm else None
        da = round(mm["AIC"] - bm["AIC"], 1) if mm else None
        l = lrt(bm, mm)
        # 方向：系数符号
        try:
            d = pd.DataFrame({"time": np.asarray(t, float), "event": np.asarray(e, int)})
            for c in X.columns:
                d[c] = np.asarray(X[c].values, float)
            cp = CoxPHFitter(penalizer=0.0).fit(d, "time", "event")
            sg = "+" if cp.params_["x"] > 0 else "-"
        except Exception:
            sg = "?"
        print(f"{name:28s} {cc:>7.4f} {dc:>+8.4f} {da:>+9.1f} {l:>13s} {sg:>6s}")
        recs.append({"construct": name, "C": cc, "dC": dc, "dAIC": da, "LRT": l})
    out[tag] = {"base": bm, "constructs": recs}
    # 复合区块：加入全部 DBP 动态
    comp = base.copy()
    comp["dbp_sd"] = z(dy["sd"]); comp["dbp_slope"] = z(dy["slope"]); comp["dbp_b40"] = z(dy["b40"])
    comp["dbp_mean"] = z(dy["mean"]); comp["dbp_min"] = z(dy["min"])
    mc = cox_metrics(t, e, comp)
    print(f"\n基准 + DBP动态区块(均值/最低/标准差/斜率/<40负荷 共5项):  C={mc['C']:.4f}  "
          f"ΔC={mc['C']-bm['C']:+.4f}  ΔAIC={mc['AIC']-bm['AIC']:+.1f}  LRT={lrt(bm,mc)}")
    out[tag]["block_dynamics"] = {"C": mc["C"], "dC": round(mc["C"] - bm["C"], 4),
                                  "dAIC": round(mc["AIC"] - bm["AIC"], 1), "LRT": lrt(bm, mc)}
    # 交互检验
    Xi = comp.copy()
    for i in range(5):
        Xi[f"vd{i}"] = comp[f"vg{i+1}"].values * z(dy["sd"])
    mi = cox_metrics(t, e, Xi)
    print(f"参考：血管药类别 × DBP不稳定 交互项块 LRT={lrt(mc, mi)}")
    return base


OUT = {}

# ---------------- MIMIC ----------------
m = pd.read_csv(os.path.join(B, "binary_mimic_data.csv"))
cc = pd.read_csv(os.path.join(B, "cont_dose_v4_mimic_data.csv"))
m = m.merge(cc[["stay_id"] + DS], on="stay_id", how="left")
m["surv"] = m["surv"].clip(lower=0.01)
y = m["mort_28d"].values.astype(int); t = m["surv"].values.astype(float)
sofa = m["sofa"].values.astype(float)
Dobs = m[D].values.astype(float)
Dbp_f = m[D].ffill(axis=1).bfill(axis=1)
dose_mean = m[DS].mean(axis=1).values
cl_vaso = gmm_cl(m[V].values, K, y)
cl_joint = gmm_cl(np.column_stack([Dbp_f.values, m[V].values]), K, y)
print(f"MIMIC n={len(m)} 28d死亡={y.mean():.1%}")
clinic(t, y, sofa, None, Dobs, dose_mean, cl_vaso, cl_joint, "MIMIC", OUT)

# ---------------- eICU ----------------
ee = pickle.load(open(os.path.join(B, "eicu_gmm_binary.pkl"), "rb"))["data"].copy()
raw = pd.read_csv(os.path.join(B, "eicu_dbp_windows_raw.csv"))
ee = ee.drop(columns=D).merge(raw, on="pid", how="left")
ee = ee[ee[D].notna().sum(axis=1) >= 4].reset_index(drop=True)
ee["surv"] = ee["los_days"].clip(lower=0.01, upper=28.0)
ee["mort"] = ee["mort"].astype(int)
mke = ee["sofa"].notna().values
ee2 = ee[mke].reset_index(drop=True)
te = ee2["surv"].values.astype(float); ye = ee2["mort"].values.astype(int)
sofe = ee2["sofa"].values.astype(float)
Dobse = ee2[D].values.astype(float)
cle_v = gmm_cl(ee2[V].values, K, ye)
print(f"\neICU n(SOFA完整)={len(ee2)}")
clinic(te, ye, sofe, None, Dobse, ee2[DS].mean(axis=1).values, cle_v, None, "eICU", OUT)

json.dump(OUT, open(os.path.join(B, "rev_dbp_role2_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print("\n[SAVED] rev_dbp_role2_results.json")
