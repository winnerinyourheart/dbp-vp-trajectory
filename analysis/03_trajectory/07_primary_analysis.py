# -*- coding: utf-8 -*-
"""
CLEP 返修：统一重跑主分析 + 新增敏感性分析
稿件 641615  Joint Trajectories of DBP and Vasopressor Requirements
产出: rev_results_v1.json + 控制台报告
"""
import sys, os, json, warnings, pickle
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import adjusted_rand_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sqlalchemy import create_engine, text

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


BASE = DATA_DIR
SEED = 42
D = [f"dbp_w{i}" for i in range(8)]
V = [f"vaso_w{i}" for i in range(8)]
R = {}   # results

M = create_engine(f"postgresql://postgres:{DB_PASSWORD}@localhost:5433/mimiciv")
E = create_engine(f"postgresql://postgres:{DB_PASSWORD}@localhost:5433/eicu")
qm = lambda s: pd.read_sql(text(s), M)
qe = lambda s: pd.read_sql(text(s), E)

# ============================================================
def load_mimic():
    m = pd.read_csv(os.path.join(BASE, "binary_mimic_data.csv"))
    m["surv"] = m["surv"].clip(lower=0.01)
    return m

def load_eicu():
    d = pickle.load(open(os.path.join(BASE, "eicu_gmm_binary.pkl"), "rb"))
    df = d["data"].copy()
    df["surv"] = df["los_days"].clip(lower=0.01, upper=28.0)
    df["mort"] = df["mort"].astype(int)
    return df

def fit_gmm_order(X, K, seed=SEED, n_init=20):
    """fit GMM, relabel classes 0..K-1 by ascending mortality"""
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    gm = GaussianMixture(n_components=K, covariance_type="full", random_state=seed,
                         n_init=n_init, max_iter=500, tol=1e-4).fit(Xs)
    cl = gm.predict(Xs)
    return gm, sc, cl

def order_by_mort(cl, y, K):
    order = sorted(range(K), key=lambda c: (y[cl == c].mean() if (cl == c).sum() else 9))
    mapper = {o: n for n, o in enumerate(order)}
    return np.array([mapper[c] for c in cl]), order

PEN_FALLBACK = []          # 记录降级：(目标惩罚, 实际惩罚)

def _fit_cox(d, pen=0.0):
    """主口径 penalizer=0（与原投稿代码一致）。若出现完全分离导致矩阵奇异，
    按 0 -> 0.01 -> 0.1 -> 0.5 降级并记录，绝不静默改变口径。"""
    chain = [pen] + [p for p in (0.01, 0.1, 0.5) if p > pen]
    last = None
    for p in chain:
        try:
            cp = CoxPHFitter(penalizer=p).fit(d, "time", "event")
            if p != pen:
                PEN_FALLBACK.append((pen, p))
            return cp
        except Exception as e:
            last = e
    raise RuntimeError(f"Cox 拟合失败（已尝试 {chain}）: {last}")

def cox_cstat(time, event, Xdf, penalizer=0.0, extra=None):
    d = pd.DataFrame({"time": np.asarray(time, float), "event": np.asarray(event, int)})
    for c in Xdf.columns:
        d[c] = np.asarray(Xdf[c].values, float)
    if extra is not None:
        d[extra.name] = np.asarray(extra.values, float)
    cp = _fit_cox(d, penalizer)
    return cp, cp.concordance_index_

def design(cl, K):
    return pd.DataFrame({f"c{c}": (cl == c).astype(float) for c in range(1, K)})

def boot_ci(time, event, Xdf, sofa, n_boot=400, seed=SEED):
    """bootstrap CI for C(class+sofa) and delta C vs sofa-only, refitting Cox each rep"""
    rng = np.random.default_rng(seed)
    n = len(time)
    time = np.asarray(time, float); event = np.asarray(event, int)
    X = Xdf.values.astype(float); s = np.asarray(sofa.values, float)
    dc, cs, c0 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            d = pd.DataFrame({"time": time[idx], "event": event[idx]})
            for j, c in enumerate(Xdf.columns):
                d[c] = X[idx, j]
            d["sofa"] = s[idx]
            cp1 = _fit_cox(d, 0.0); a = cp1.concordance_index_
            cp0 = _fit_cox(d[["time", "event", "sofa"]], 0.0); b = cp0.concordance_index_
            cs.append(a); c0.append(b); dc.append(a - b)
        except Exception:
            continue
    pct = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"c_class_sofa_ci": pct(cs), "c_sofa_ci": pct(c0), "dC_ci": pct(dc), "n_boot_ok": len(cs)}

def landmark(time, event, cl, sofa, K, lm=2.0, boot=True):
    """landmark at lm days: keep patients with time > lm, reset clock"""
    time = np.asarray(time, float); event = np.asarray(event, int); cl = np.asarray(cl)
    sofa = np.asarray(sofa, float)
    keep = time > lm
    t2 = time[keep] - lm
    e2 = event[keep]; c2 = cl[keep]; s2 = sofa[keep]
    X = design(c2, K)
    cp1, c1 = cox_cstat(t2, e2, X, penalizer=0.0, extra=pd.Series(s2, name="sofa"))
    cp0, c0 = cox_cstat(t2, e2, pd.DataFrame(index=range(len(t2))), penalizer=0.0,
                        extra=pd.Series(s2, name="sofa"))
    cpU, cU = cox_cstat(t2, e2, X, penalizer=0.0)
    out = {"n_excluded": int((~keep).sum()), "pct_excluded": round(float((~keep).mean() * 100), 1),
           "n_landmark": int(keep.sum()), "c_class_only": round(cU, 4),
           "c_sofa": round(c0, 4), "c_class_sofa": round(c1, 4), "dC": round(c1 - c0, 4)}
    if boot:
        out.update(boot_ci(t2, e2, X, pd.Series(s2), n_boot=300))
    return out

# ============================================================
print("#" * 78); print("# [0] 数据载入"); print("#" * 78)
mm = load_mimic(); ee = load_eicu()
print(f"MIMIC n={len(mm)}  mort={mm.mort_28d.mean():.4f}  SOFA={mm.sofa.mean():.2f}")
print(f"eICU  n={len(ee)}  mort={ee.mort.mean():.4f}   SOFA={ee.sofa.dropna().mean():.2f} (missing {ee.sofa.isna().sum()})")

miss = mm[D].isna().mean().round(4)
print("\nMIMIC DBP 各窗口缺失率:"); print(miss.to_string())
miss_e = ee[D].isna().mean().round(4)
print("\neICU DBP 各窗口缺失率:"); print(miss_e.to_string())
R["missing_rates"] = {"mimic": {k: float(v) for k, v in miss.items()},
                      "eicu": {k: float(v) for k, v in miss_e.items()}}

# pkl 中的 eICU DBP 已填补 -> 直接查库重算原始窗口缺失率
_pids = ee["pid"].astype(int).tolist()
_rows = []
for _i in range(0, len(_pids), 800):
    _ch = ",".join(str(x) for x in _pids[_i:_i + 800])
    _rows.append(qe(f"""SELECT patientunitstayid, FLOOR(chartoffset/360)::int w,
      AVG(COALESCE(ibp_diastolic,nibp_diastolic)) dbp
    FROM eicu_crd.pivoted_vital
    WHERE patientunitstayid IN ({_ch}) AND COALESCE(ibp_diastolic,nibp_diastolic) IS NOT NULL
      AND chartoffset>=0 AND chartoffset<2880
    GROUP BY patientunitstayid, FLOOR(chartoffset/360)::int"""))
_raw = pd.concat(_rows, ignore_index=True)
_raw = _raw[_raw["w"].between(0, 7)]
_piv = _raw.pivot_table(index="patientunitstayid", columns="w", values="dbp").reindex(columns=range(8)).reindex(_pids)
_raw_miss = _piv.isna().mean().round(4)
print("\neICU 原始 DBP 各窗口缺失率 (直接查库, 未填补):"); print(_raw_miss.to_string())
R["missing_rates"]["eicu_raw_db"] = {f"dbp_w{k}": float(v) for k, v in _raw_miss.items()}
R["eicu_raw_complete_case_n"] = int(_piv.notna().all(axis=1).sum())
# 用原始 DBP 替换 pkl 中已填补的 DBP, 全流程统一为 LOCF
_eicu_db = _piv.copy(); _eicu_db.columns = D
_eicu_db = _eicu_db.reset_index().rename(columns={"patientunitstayid": "pid"})
_eicu_db.to_csv(os.path.join(BASE, "eicu_dbp_windows_raw.csv"), index=False)
_n_before = len(ee)
ee = ee.drop(columns=D).merge(_eicu_db, on="pid", how="left")
ee = ee[ee[D].notna().sum(axis=1) >= 4].reset_index(drop=True)
print(f"eICU 采用原始 DBP (未填补) 并施加 n>=4 规则: {_n_before} -> {len(ee)}")
R["eicu_n_after_raw_dbp_rule"] = int(len(ee))

# ---- 按窗口缺失模式: 前向可填补比例
m_fwd = mm[D].ffill(axis=1).isna().mean().round(4)
print("\nLOCF(仅前向) 后残余缺失率:"); print(m_fwd.to_string())
print("LOCF+后向 后残余缺失率:", float(mm[D].ffill(axis=1).bfill(axis=1).isna().mean().mean()))
R["locf_residual"] = {"forward_only": {k: float(v) for k, v in m_fwd.items()},
                      "forward_backward_mean": float(mm[D].ffill(axis=1).bfill(axis=1).isna().mean().mean())}

# ============================================================
print("\n" + "#" * 78); print("# [1] 主分析 (缺失值 = LOCF 前向+后向)"); print("#" * 78)
Db = mm[D].ffill(axis=1).bfill(axis=1)          # LOCF primary
Xb = np.column_stack([Db.values, mm[V].values])
gm_b, sc_b, cl_raw = fit_gmm_order(Xb, 6)
y = mm["mort_28d"].values.astype(int)
cl_b, _ = order_by_mort(cl_raw, y, 6)
mm["class_rev"] = cl_b

tab = pd.DataFrame({"class": range(6)})
tab["n"] = [int((cl_b == c).sum()) for c in range(6)]
tab["pct"] = (tab["n"] / len(mm) * 100).round(1)
tab["mort"] = [round(float(y[cl_b == c].mean()), 4) for c in range(6)]
tab["sofa"] = [round(float(mm.sofa[cl_b == c].mean()), 2) for c in range(6)]
print(tab.to_string(index=False))
R["primary_mimic_class"] = tab.to_dict("records")

sofa = mm["sofa"].values.astype(float)
surv = mm["surv"].values.astype(float)
Xc = design(cl_b, 6)
cpU, cU = cox_cstat(surv, y, Xc, penalizer=0.0)
cpS, cS = cox_cstat(surv, y, pd.DataFrame(index=range(len(y))), penalizer=0.0, extra=pd.Series(sofa, name="sofa"))
cpA, cA = cox_cstat(surv, y, Xc, penalizer=0.0, extra=pd.Series(sofa, name="sofa"))
print(f"\nC(class only)={cU:.4f}  C(SOFA only)={cS:.4f}  C(class+SOFA)={cA:.4f}  dC={cA-cS:+.4f}")
R["primary_mimic_cstat"] = {"class_only": round(cU, 4), "sofa_only": round(cS, 4),
                            "class_sofa": round(cA, 4), "delta": round(cA - cS, 4)}
print("bootstrap CI ...")
bci = boot_ci(surv, y, Xc, pd.Series(sofa, name="sofa"), n_boot=400)
print("  ", bci)
R["primary_mimic_boot"] = bci

# 未调整 HR
print("\n未调整 HR (vs C0):")
hrs = {}
for c in range(1, 6):
    row = cpU.summary.loc[f"c{c}"]
    lo = np.exp(row["coef"] - 1.96 * row["se(coef)"]); hi = np.exp(row["coef"] + 1.96 * row["se(coef)"])
    hrs[f"C{c}"] = {"HR": round(float(row["exp(coef)"]), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2)}
    print(f"  C{c}: HR {row['exp(coef)']:.2f} ({lo:.2f}-{hi:.2f})")
R["primary_mimic_hr"] = hrs

# ============================================================
print("\n" + "#" * 78); print("# [2] 缺失数据敏感性分析 (MIMIC)"); print("#" * 78)
variants = {}
variants["LOCF (前向+后向)"] = np.column_stack([Db.values, mm[V].values])

cc = mm[D].notna().all(axis=1).values
variants[f"完整病例 (n={int(cc.sum())})"] = np.column_stack([mm.loc[cc, D].values, mm.loc[cc, V].values])
# 原投稿方案对照: 缺失 DBP 置 0 (生理上不可能), 用于说明本次修正的影响
variants["零值填补 (原投稿方案, 对照)"] = np.column_stack([mm[D].fillna(0).values, mm[V].values])

# 多重插补
mi_cols = D + V + ["sofa"]
mi_X = mm[mi_cols].copy()
imp_ari, imp_c, imp_dc, imp_mort = [], [], [], []
for s in range(42, 47):
    it = IterativeImputer(estimator=BayesianRidge(),
                          max_iter=10, random_state=s, sample_posterior=True)
    Xi = it.fit_transform(mi_X)
    Xm = np.column_stack([Xi[:, :8], Xi[:, 8:16]])
    _, _, cl_i = fit_gmm_order(Xm, 6)
    cl_i, _ = order_by_mort(cl_i, y, 6)
    imp_ari.append(adjusted_rand_score(cl_b, cl_i))
    Xci = design(cl_i, 6)
    _, c1 = cox_cstat(surv, y, Xci, penalizer=0.0, extra=pd.Series(sofa, name="sofa"))
    _, c0 = cox_cstat(surv, y, pd.DataFrame(index=range(len(y))), penalizer=0.0, extra=pd.Series(sofa, name="sofa"))
    imp_c.append(c1); imp_dc.append(c1 - c0)
    imp_mort.append([float(y[cl_i == c].mean()) for c in range(6)])
print(f"MI 5 套: ARI(vs主分析)={np.mean(imp_ari):.3f}±{np.std(imp_ari):.3f}")
print(f"         C(class+SOFA)={np.mean(imp_c):.4f}±{np.std(imp_c):.4f}   dC={np.mean(imp_dc):+.4f}±{np.std(imp_dc):.4f}")
print(f"         各类死亡率均值={np.round(np.mean(imp_mort,axis=0),4).tolist()}")
R["mi"] = {"ari_vs_primary": [round(float(np.mean(imp_ari)), 3), round(float(np.std(imp_ari)), 3)],
           "c_class_sofa": [round(float(np.mean(imp_c)), 4), round(float(np.std(imp_c)), 4)],
           "dC": [round(float(np.mean(imp_dc)), 4), round(float(np.std(imp_dc)), 4)],
           "class_mort": [round(float(x), 4) for x in np.mean(imp_mort, axis=0)]}

sens = {}
for name, X in variants.items():
    _, _, clv = fit_gmm_order(X, 6)
    n_used = len(X)
    yy = y if n_used == len(mm) else y[cc]
    sv = surv if n_used == len(mm) else surv[cc]
    sf = sofa if n_used == len(mm) else sofa[cc]
    clv, _ = order_by_mort(clv, yy, 6)
    Xcv = design(clv, 6)
    _, c1 = cox_cstat(sv, yy, Xcv, penalizer=0.0, extra=pd.Series(sf, name="sofa"))
    _, c0 = cox_cstat(sv, yy, pd.DataFrame(index=range(len(sv))), penalizer=0.0, extra=pd.Series(sf, name="sofa"))
    ari = adjusted_rand_score(np.asarray(cl_b)[:len(clv)] if n_used == len(mm) else cl_b[cc], clv)
    morts = [float(yy[clv == c].mean()) for c in range(6)]
    sens[name] = {"n": int(n_used), "ari_vs_primary": round(float(ari), 3),
                  "mort_gradient": [round(m, 4) for m in morts],
                  "c_sofa": round(c0, 4), "c_class_sofa": round(c1, 4), "dC": round(c1 - c0, 4)}
    print(f"\n{name}: n={n_used}  ARI={ari:.3f}  死亡率={min(morts):.1%}–{max(morts):.1%}")
    print(f"    C(SOFA)={c0:.4f}  C(class+SOFA)={c1:.4f}  dC={c1-c0:+.4f}")
R["missing_sensitivity"] = sens

# ============================================================
print("\n" + "#" * 78); print("# [3] 48h Landmark 分析"); print("#" * 78)
R["landmark_mimic"] = landmark(surv, y, cl_b, sofa, 6, lm=2.0)
print("MIMIC 48h landmark:", json.dumps(R["landmark_mimic"], ensure_ascii=False))

# eICU
ee2 = ee[ee["sofa"].notna()].reset_index(drop=True)
Xe = np.column_stack([ee2[D].ffill(axis=1).bfill(axis=1).values, ee2[V].values])
_, _, ecl_raw = fit_gmm_order(Xe, 6)
ye = ee2["mort"].values.astype(int)
ecl, _ = order_by_mort(ecl_raw, ye, 6)
ee2["class_rev"] = ecl
tab_e = pd.DataFrame({"class": range(6)})
tab_e["n"] = [int((ecl == c).sum()) for c in range(6)]
tab_e["mort"] = [round(float(ye[ecl == c].mean()), 4) for c in range(6)]
tab_e["sofa"] = [round(float(ee2.sofa[ecl == c].mean()), 2) for c in range(6)]
print("\neICU 独立 GMM (完整 SOFA 子集):"); print(tab_e.to_string(index=False))
R["primary_eicu_class"] = tab_e.to_dict("records")
R["primary_eicu_n_sofa_complete"] = int(len(ee2))

Xce = design(ecl, 6)
se = ee2["surv"].values.astype(float); sfe = ee2["sofa"].values.astype(float)
_, ecS = cox_cstat(se, ye, pd.DataFrame(index=range(len(ye))), penalizer=0.0, extra=pd.Series(sfe, name="sofa"))
_, ecA = cox_cstat(se, ye, Xce, penalizer=0.0, extra=pd.Series(sfe, name="sofa"))
_, ecU = cox_cstat(se, ye, Xce, penalizer=0.0)
print(f"\neICU C(class only)={ecU:.4f}  C(SOFA)={ecS:.4f}  C(class+SOFA)={ecA:.4f}  dC={ecA-ecS:+.4f}")
R["primary_eicu_cstat"] = {"class_only": round(ecU, 4), "sofa_only": round(ecS, 4),
                           "class_sofa": round(ecA, 4), "delta": round(ecA - ecS, 4)}
eb = boot_ci(se, ye, Xce, pd.Series(sfe, name="sofa"), n_boot=300)
print("  bootstrap:", eb); R["primary_eicu_boot"] = eb

# eICU landmark (全样本，用完整SOFA子集)
full_e = ee.copy()
Xef = np.column_stack([full_e[D].ffill(axis=1).bfill(axis=1).values, full_e[V].values])
_, _, ef_raw = fit_gmm_order(Xef, 6)
yef = full_e["mort"].values.astype(int)
efl, _ = order_by_mort(ef_raw, yef, 6)
print("\neICU 48h landmark (全样本, SOFA 缺失以中位数填补用于 Cox):")
R["landmark_eicu"] = landmark(full_e["surv"].values, yef, efl, full_e["sofa"].fillna(full_e["sofa"].median()).values, 6, lm=2.0)
print("  ", json.dumps(R["landmark_eicu"], ensure_ascii=False))

# ============================================================
print("\n" + "#" * 78); print("# [4] K 值敏感性分析 (MIMIC, LOCF)"); print("#" * 78)
ks = {}
for K in [5, 6, 7]:
    _, _, clk = fit_gmm_order(Xb, K)
    clk, _ = order_by_mort(clk, y, K)
    Xck = design(clk, K)
    _, c1 = cox_cstat(surv, y, Xck, penalizer=0.0, extra=pd.Series(sofa, name="sofa"))
    _, c0 = cox_cstat(surv, y, pd.DataFrame(index=range(len(y))), penalizer=0.0, extra=pd.Series(sofa, name="sofa"))
    _, cu = cox_cstat(surv, y, Xck, penalizer=0.0)
    morts = [float(y[clk == c].mean()) for c in range(K)]
    sizes = [int((clk == c).sum()) for c in range(K)]
    ari6 = adjusted_rand_score(cl_b, clk) if K == 6 else None
    ks[K] = {"c_class_only": round(cu, 4), "c_sofa": round(c0, 4), "c_class_sofa": round(c1, 4),
             "dC": round(c1 - c0, 4), "mort_min": round(min(morts), 4), "mort_max": round(max(morts), 4),
             "mort_gradient_width": round(max(morts) - min(morts), 4), "min_class_size": min(sizes),
             "sizes": sizes}
    print(f"K={K}: C(class)={cu:.4f}  C(SOFA)={c0:.4f}  C(class+SOFA)={c1:.4f}  dC={c1-c0:+.4f}  "
          f"死亡 {min(morts):.1%}-{max(morts):.1%}  最小类 n={min(sizes)}")

# K=5/7 vs K=6 的 ARI（评估类别稳定性）
for K in [5, 7]:
    _, _, clk = fit_gmm_order(Xb, K)
    print(f"  ARI(K={K} vs K=6) = {adjusted_rand_score(cl_b, clk):.3f}")
    ks[K]["ari_vs_k6"] = round(float(adjusted_rand_score(cl_b, clk)), 3)
R["k_sensitivity"] = ks
R["bic_series"] = json.load(open(os.path.join(BASE, "bic_series_binary.json")))

# ============================================================
print("\n" + "#" * 78); print("# [5] 连续剂量 vs 二分类 (MIMIC)"); print("#" * 78)
cd = pd.read_csv(os.path.join(BASE, "cont_dose_v4_mimic_data.csv"))
cd = cd[cd["stay_id"].isin(mm["stay_id"])].reset_index(drop=True)
Dc = [f"dose_w{i}" for i in range(8)]
cdd = cd[D].ffill(axis=1).bfill(axis=1)
dose = cd[Dc].copy()
# ln(AUC+eps) 已在 pipeline 中，直接标准化
Xcd = np.column_stack([cdd.values, dose.values])
_, _, cc_raw = fit_gmm_order(Xcd, 6)
yc = cd["mort_28d"].values.astype(int)
ccl, _ = order_by_mort(cc_raw, yc, 6)
Xcc = design(ccl, 6)
_, cc1 = cox_cstat(cd["surv"].clip(lower=0.01).values, yc, Xcc, penalizer=0.0,
                   extra=pd.Series(cd["sofa"].values, name="sofa"))
_, cc0 = cox_cstat(cd["surv"].clip(lower=0.01).values, yc, pd.DataFrame(index=range(len(cd))), penalizer=0.0,
                   extra=pd.Series(cd["sofa"].values, name="sofa"))
_, ccu = cox_cstat(cd["surv"].clip(lower=0.01).values, yc, Xcc, penalizer=0.0)
morts_c = [float(yc[ccl == c].mean()) for c in range(6)]
ari_cd = adjusted_rand_score(mm["class_rev"].values[:len(ccl)], ccl)
print(f"连续 NE 剂量 K=6: n={len(cd)}  C(class)={ccu:.4f}  C(SOFA)={cc0:.4f}  C(class+SOFA)={cc1:.4f}  dC={cc1-cc0:+.4f}")
print(f"  死亡率范围 {min(morts_c):.1%}-{max(morts_c):.1%}   ARI(vs 二分类)={ari_cd:.3f}")
R["continuous_dose"] = {"n": int(len(cd)), "c_class_only": round(ccu, 4), "c_sofa": round(cc0, 4),
                        "c_class_sofa": round(cc1, 4), "dC": round(cc1 - cc0, 4),
                        "mort_min": round(min(morts_c), 4), "mort_max": round(max(morts_c), 4),
                        "ari_vs_binary": round(float(ari_cd), 3)}

# ============================================================
print("\n" + "#" * 78); print("# [6] 血管活性药使用梯度 (MIMIC)"); print("#" * 78)
ag = qm("""
SELECT ie.stay_id,
 MAX(CASE WHEN LOWER(ie.itemid::text) IS NOT NULL AND LOWER(dl.label) LIKE '%norepinephrine%' THEN 1 ELSE 0 END) ne,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%vasopressin%' THEN 1 ELSE 0 END) vp,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%epinephrine%'
               AND LOWER(dl.label) NOT LIKE '%norepinephrine%'
               AND LOWER(dl.label) NOT LIKE '%phenylephrine%' THEN 1 ELSE 0 END) epi,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%phenylephrine%' THEN 1 ELSE 0 END) pe,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%dopamine%' THEN 1 ELSE 0 END) da,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%dobutamine%' THEN 1 ELSE 0 END) dob,
 MAX(CASE WHEN LOWER(dl.label) LIKE '%milrinone%' THEN 1 ELSE 0 END) mil
FROM mimiciv_icu.inputevents ie
JOIN mimiciv_icu.d_items dl ON ie.itemid = dl.itemid
WHERE ie.stay_id IN (SELECT DISTINCT stay_id FROM public.ds_mimic_shock)
  AND (LOWER(dl.label) LIKE '%norepinephrine%' OR LOWER(dl.label) LIKE '%vasopressin%'
    OR LOWER(dl.label) LIKE '%epinephrine%' OR LOWER(dl.label) LIKE '%phenylephrine%'
    OR LOWER(dl.label) LIKE '%dopamine%' OR LOWER(dl.label) LIKE '%dobutamine%'
    OR LOWER(dl.label) LIKE '%milrinone%')
GROUP BY ie.stay_id
""")
mg = mm[["stay_id"]].merge(ag, on="stay_id", how="left")
for c2 in ["ne", "vp", "epi", "pe", "da", "dob", "mil"]:
    mg[c2] = mg[c2].fillna(0).astype(int)
mg["extra"] = mg[["vp", "epi", "pe", "da", "dob", "mil"]].sum(axis=1)
mg["two_extra"] = (mg["extra"] >= 2).astype(int)
mg["class"] = cl_b
tab6 = mg.groupby("class")[["ne", "vp", "epi", "two_extra"]].mean().round(4) * 100
tab6["n"] = mg.groupby("class").size()
print(tab6.to_string())
R["agent_table_mimic"] = tab6.reset_index().to_dict("records")

print("\n--- eICU 用药梯度 ---")
pids_e = full_e["pid"].astype(int).tolist()
_rows2 = []
for _i in range(0, len(pids_e), 800):
    _ch = ",".join(str(x) for x in pids_e[_i:_i + 800])
    _rows2.append(qe(f"""SELECT patientunitstayid,
      MAX(CASE WHEN drugname ILIKE '%norepinephrine%' OR drugname ILIKE '%noradrenaline%' THEN 1 ELSE 0 END) ne,
      MAX(CASE WHEN drugname ILIKE '%vasopressin%' THEN 1 ELSE 0 END) vp,
      MAX(CASE WHEN (drugname ILIKE '%epinephrine%' OR drugname ILIKE '%adrenaline%')
                AND drugname NOT ILIKE '%phenylephrine%'
                AND drugname NOT ILIKE '%norepinephrine%'
                AND drugname NOT ILIKE '%noradrenaline%' THEN 1 ELSE 0 END) epi,
      MAX(CASE WHEN drugname ILIKE '%phenylephrine%' THEN 1 ELSE 0 END) pe,
      MAX(CASE WHEN drugname ILIKE '%dopamine%' THEN 1 ELSE 0 END) da,
      MAX(CASE WHEN drugname ILIKE '%dobutamine%' THEN 1 ELSE 0 END) dob,
      MAX(CASE WHEN drugname ILIKE '%milrinone%' THEN 1 ELSE 0 END) mil
    FROM eicu_crd.infusiondrug
    WHERE patientunitstayid IN ({_ch}) GROUP BY patientunitstayid"""))
age = pd.concat(_rows2, ignore_index=True)
me = full_e[["pid"]].rename(columns={"pid": "patientunitstayid"}).merge(age, on="patientunitstayid", how="left")
for c2 in ["ne", "vp", "epi", "pe", "da", "dob", "mil"]:
    me[c2] = me[c2].fillna(0).astype(int)
me["extra"] = me[["vp", "epi", "pe", "da", "dob", "mil"]].sum(axis=1)
me["two_extra"] = (me["extra"] >= 2).astype(int)
me["class"] = efl
tab6e = me.groupby("class")[["ne", "vp", "epi", "two_extra"]].mean().round(4) * 100
tab6e["n"] = me.groupby("class").size()
print(tab6e.to_string())
R["agent_table_eicu"] = tab6e.reset_index().to_dict("records")

print("\n" + "#" * 78); print("# [7] 结局 95% CI (分类死亡率, Wilson)"); print("#" * 78)
def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round((c - h) * 100, 1), round((c + h) * 100, 1)
mort_ci = []
for c in range(6):
    k = int(y[cl_b == c].sum()); n = int((cl_b == c).sum())
    lo, hi = wilson(k, n)
    mort_ci.append({"class": f"C{c}", "n": n, "events": k, "mort_pct": round(k / n * 100, 1), "ci": [lo, hi]})
    print(f"  C{c}: {k}/{n} = {k/n*100:.1f}% (95%CI {lo}-{hi})")
R["mimic_class_mort_ci"] = mort_ci

mort_ci_e = []
for c in range(6):
    k = int(ye[ecl == c].sum()); n = int((ecl == c).sum())
    lo, hi = wilson(k, n)
    mort_ci_e.append({"class": f"C{c}", "n": n, "events": k, "mort_pct": round(k / n * 100, 1), "ci": [lo, hi]})
print("\neICU:")
for r0 in mort_ci_e:
    print(f"  {r0['class']}: {r0['events']}/{r0['n']} = {r0['mort_pct']}% (95%CI {r0['ci'][0]}-{r0['ci'][1]})")
R["eicu_class_mort_ci"] = mort_ci_e

with open(os.path.join(BASE, "rev_results_v1.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(f"\n[口径] 主分析 Cox penalizer=0.00；降级次数={len(PEN_FALLBACK)}")
if PEN_FALLBACK:
    print("  降级明细:", PEN_FALLBACK[:20])
R["pen_fallback_count"] = len(PEN_FALLBACK)
R["pen_fallback_detail"] = [list(map(float, x)) for x in PEN_FALLBACK[:50]]
json.dump(R, open(os.path.join(BASE, "rev_results_v1.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("[SAVED] rev_results_v1.json")
print("[DONE]")
