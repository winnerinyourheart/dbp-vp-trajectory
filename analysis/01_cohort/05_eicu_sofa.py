# -*- coding: utf-8 -*-
"""
eICU-CRD SOFA 重建脚本（找回缺失的推导代码）
—— 按标准 SOFA 定义，从 eicu_crd 源表计算入 ICU 后首个 24 小时的最差分项之和。
—— 计算完成后与 eicu_gmm_binary.pkl 中已有的 sofa 值做一致性核验。
"""
import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


B = DATA_DIR
E = create_engine(f"postgresql://postgres:{DB_PASSWORD}@localhost:5433/eicu")
WIN = "between 0 and 1440"          # 入 ICU 后 24 小时

pids = pickle.load(open(os.path.join(B, "eicu_gmm_binary.pkl"), "rb"))["data"]["pid"].astype(int).tolist()
print(f"目标队列 n = {len(pids)}")


def q(sql, chunk=900):
    out = []
    for i in range(0, len(pids), chunk):
        ch = ",".join(map(str, pids[i:i + chunk]))
        out.append(pd.read_sql(text(sql.format(ch=ch)), E))
    return pd.concat(out, ignore_index=True)


# ---------- 1. 实验室指标 ----------
lab = q(f"""SELECT patientunitstayid, labname, labresult
FROM eicu_crd.lab
WHERE labresultoffset {WIN} AND patientunitstayid IN ({{ch}})
  AND labname IN ('platelets x 1000','total bilirubin','creatinine','paO2')""")
lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")
lab = lab.dropna(subset=["labresult"])
pl = lab[lab.labname == "platelets x 1000"].groupby("patientunitstayid").labresult.min()
bi = lab[lab.labname == "total bilirubin"].groupby("patientunitstayid").labresult.max()
cr = lab[lab.labname == "creatinine"].groupby("patientunitstayid").labresult.max()
po = lab[lab.labname == "paO2"].groupby("patientunitstayid").labresult.min()

# ---------- 2. 生命体征（MAP 最低值）----------
vp = q(f"""SELECT patientunitstayid, MIN(systemicmean) AS map_min
FROM eicu_crd.vitalperiodic
WHERE observationoffset {WIN} AND patientunitstayid IN ({{ch}}) AND systemicmean > 0
GROUP BY patientunitstayid""").set_index("patientunitstayid").map_min

# ---------- 3. GCS 最低值 ----------
gc = q(f"""SELECT patientunitstayid, MIN(NULLIF(regexp_replace(nursingchartvalue,'[^0-9]','','g'),'')::numeric) AS gcs_min
FROM eicu_crd.nursecharting
WHERE nursingchartoffset {WIN} AND patientunitstayid IN ({{ch}})
  AND nursingchartcelltypevalname = 'GCS Total'
GROUP BY patientunitstayid""").set_index("patientunitstayid").gcs_min

# ---------- 4. 尿量（24h 合计，mL）----------
uo = q(f"""SELECT patientunitstayid, SUM(cellvaluenumeric) AS uo
FROM eicu_crd.intakeoutput
WHERE intakeoutputoffset {WIN} AND patientunitstayid IN ({{ch}})
  AND celllabel ILIKE '%urine%' AND cellvaluenumeric IS NOT NULL
GROUP BY patientunitstayid""").set_index("patientunitstayid").uo

# ---------- 5. 呼吸支持 ----------
vent = q(f"""SELECT DISTINCT patientunitstayid
FROM eicu_crd.respiratorycare
WHERE respcarestatusoffset {WIN} AND patientunitstayid IN ({{ch}})
  AND airwaytype IS NOT NULL AND airwaytype NOT IN ('None','none','')""").patientunitstayid
fio = q(f"""SELECT patientunitstayid, MAX(NULLIF(regexp_replace(respchartvalue,'[^0-9\\.]','','g'),'')::numeric) AS fio2_max
FROM eicu_crd.respiratorycharting
WHERE respchartoffset {WIN} AND patientunitstayid IN ({{ch}})
  AND respchartvaluelabel IN ('FiO2','FIO2 (%)','Set Fraction of Inspired Oxygen (FIO2)')
  AND respchartvalue ~ '^[0-9\\.]+$'
GROUP BY patientunitstayid""").set_index("patientunitstayid").fio2_max

# ---------- 6. 升压药最大速率（统一为 NE 当量的近似处理）----------
iv = q(f"""SELECT patientunitstayid, drugname,
       MAX(NULLIF(regexp_replace(drugrate,'[^0-9\\.]','','g'),'')::numeric) AS rate
FROM eicu_crd.infusiondrug
WHERE infusionoffset {WIN} AND patientunitstayid IN ({{ch}})
GROUP BY patientunitstayid, drugname""")
wt = q(f"""SELECT patientunitstayid,
       MAX(NULLIF(regexp_replace(patientweight,'[^0-9\\.]','','g'),'')::numeric) AS wt
FROM eicu_crd.infusiondrug
WHERE infusionoffset {WIN} AND patientunitstayid IN ({{ch}})
GROUP BY patientunitstayid""").set_index("patientunitstayid").wt
wt2 = q(f"""SELECT patientunitstayid, MAX(admissionweight) AS wt
FROM eicu_crd.patient
WHERE patientunitstayid IN ({{ch}}) AND admissionweight > 0
GROUP BY patientunitstayid""").set_index("patientunitstayid").wt
wt = pd.concat([wt.reindex(pids), wt2.reindex(pids)], axis=1).max(axis=1)


def drug_max(pat):
    s = iv[iv.drugname.str.lower().str.contains(pat, regex=True)]
    return s.groupby("patientunitstayid").rate.max()


ne, dop, dob, epi = drug_max("norepinephrine"), drug_max("dopamine"), drug_max("dobutamine"), drug_max("epinephrine")

# 单位处理：mcg/kg/min 直接使用；mcg/min 除以体重；ml/hr 按 4 mg/250 mL（16 mcg/mL）折算后除以体重
ne_kg = iv[iv.drugname.str.lower().str.contains("norepinephrine")]
ne_mcgkg = ne_kg[ne_kg.drugname.str.contains("mcg/kg/min")].groupby("patientunitstayid").rate.max()
ne_mcgmin = ne_kg[ne_kg.drugname.str.contains("mcg/min")].groupby("patientunitstayid").rate.max()
ne_mlhr = ne_kg[ne_kg.drugname.str.contains("ml/hr", case=False)].groupby("patientunitstayid").rate.max()


def to_mcgkg(min_series, ml_series):
    w = wt.reindex(pids)
    a = (min_series.reindex(pids) / w)
    b = (ml_series.reindex(pids) * 16.0 / 60.0) / w        # ml/hr -> mcg/min -> mcg/kg/min
    return pd.concat([a, b], axis=1).max(axis=1)


ne_final = pd.concat([ne_mcgkg.reindex(pids), to_mcgkg(ne_mcgmin, ne_mlhr)], axis=1).max(axis=1)

df = pd.DataFrame(index=pd.Index(pids, name="patientunitstayid"))
df["plt"], df["bil"], df["cre"], df["pao2"] = pl.reindex(pids), bi.reindex(pids), cr.reindex(pids), po.reindex(pids)
df["mapmin"], df["gcs"], df["uo"] = vp.reindex(pids), gc.reindex(pids), uo.reindex(pids)
df["fio2"] = fio.reindex(pids)
df["vent"] = df.index.isin(set(vent))
df["nekg"], df["dop"], df["dob"], df["epi"] = ne_final, dop.reindex(pids), dob.reindex(pids), epi.reindex(pids)


def s_resp(r):
    if pd.isna(r.pao2) or pd.isna(r.fio2) or r.fio2 <= 0:
        return np.nan
    f = r.fio2 / 100.0 if r.fio2 > 1 else r.fio2
    if f <= 0:
        return np.nan
    pf = r.pao2 / f
    if pf < 100 and r.vent: return 4
    if pf < 200 and r.vent: return 3
    if pf < 300: return 2
    if pf < 400: return 1
    return 0


def s_coag(x): return np.nan if pd.isna(x) else (4 if x < 20 else 3 if x < 50 else 2 if x < 100 else 1 if x < 150 else 0)
def s_liver(x): return np.nan if pd.isna(x) else (4 if x >= 12 else 3 if x >= 6 else 2 if x >= 2 else 1 if x >= 1.2 else 0)


def s_cardio(r):
    if pd.notna(r.dop) and r.dop > 15: return 4
    if pd.notna(r.epi) and r.epi > 0.1: return 4
    if pd.notna(r.nekg) and r.nekg > 0.1: return 4
    if pd.notna(r.dop) and r.dop > 5: return 3
    if pd.notna(r.dop) and r.dop > 0: return 2
    if pd.notna(r.dob) and r.dob > 0: return 2
    if pd.notna(r.nekg) and r.nekg > 0: return 3
    if pd.notna(r.epi) and r.epi > 0: return 3
    if pd.notna(r.mapmin) and r.mapmin < 70: return 1
    return 0 if pd.notna(r.mapmin) else np.nan


def s_cns(x): return np.nan if pd.isna(x) else (4 if x < 6 else 3 if x < 10 else 2 if x < 13 else 1 if x < 15 else 0)


def s_renal(r):
    cre = np.nan
    if pd.notna(r.cre):
        cre = 4 if r.cre >= 5 else 3 if r.cre >= 3.5 else 2 if r.cre >= 2 else 1 if r.cre >= 1.2 else 0
    u = np.nan
    if pd.notna(r.uo):
        u = 4 if r.uo < 200 else 3 if r.uo < 500 else 0
    if pd.isna(cre): return u
    if pd.isna(u): return cre
    return max(cre, u)


df["s_resp"] = df.apply(s_resp, axis=1)
df["s_coag"] = df.plt.apply(s_coag)
df["s_liver"] = df.bil.apply(s_liver)
df["s_cardio"] = df.apply(s_cardio, axis=1)
df["s_cns"] = df.gcs.apply(s_cns)
df["s_renal"] = df.apply(s_renal, axis=1)
COMP = ["s_resp", "s_coag", "s_liver", "s_cardio", "s_cns", "s_renal"]
df["sofa_derived"] = df[COMP].sum(axis=1, skipna=True)
df.loc[df[COMP].isna().all(axis=1), "sofa_derived"] = np.nan

print("\n=== 分项可得率 ===")
for c in COMP:
    print(f"  {c:9s} 可得 {df[c].notna().sum():5d} / {len(df)}  均值 {df[c].mean():.2f}")
print(f"\n重建 SOFA：可得 {df.sofa_derived.notna().sum()}，均值 {df.sofa_derived.mean():.2f}，"
      f"中位 {df.sofa_derived.median():.0f}，范围 {df.sofa_derived.min():.0f}–{df.sofa_derived.max():.0f}")

old = pickle.load(open(os.path.join(B, "eicu_gmm_binary.pkl"), "rb"))["data"][["pid", "sofa"]].rename(
    columns={"pid": "patientunitstayid", "sofa": "sofa_stored"}).set_index("patientunitstayid")
cmp = df[["sofa_derived"]].join(old)
both = cmp.dropna()
print(f"\n=== 与存量 sofa 的一致性（双方均非缺失 n={len(both)}）===")
print(f"  完全一致       : {int((both.sofa_derived == both.sofa_stored).sum())} ({100*(both.sofa_derived==both.sofa_stored).mean():.1f}%)")
print(f"  相差 ≤1 分     : {int((abs(both.sofa_derived-both.sofa_stored) <= 1).sum())} ({100*(abs(both.sofa_derived-both.sofa_stored)<=1).mean():.1f}%)")
print(f"  相关系数 r     : {both.corr().iloc[0,1]:.3f}")
print(f"  平均差（重建−存量）: {(both.sofa_derived-both.sofa_stored).mean():+.2f}")
print(f"  存量缺失而重建可得: {int(cmp.sofa_stored.isna().sum() - (cmp.sofa_stored.isna() & cmp.sofa_derived.isna()).sum())}")

os.makedirs(os.path.join(B, "recovered"), exist_ok=True)
df.reset_index()[["patientunitstayid"] + COMP + ["sofa_derived"]].to_csv(
    os.path.join(B, "recovered", "eicu_sofa_derived.csv"), index=False)
print("\n[SAVED] recovered/eicu_sofa_derived.csv")
