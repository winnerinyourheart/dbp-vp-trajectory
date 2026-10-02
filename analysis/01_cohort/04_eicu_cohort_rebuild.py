"""
Full eICU cohort re-extraction with correct SOFA
Starting from ds_eicu_shock (pre-built Sepsis-3 shock)
"""
import os
import pg8000, pandas as pd, numpy as np

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


conn = pg8000.connect(host='localhost', port=5433, user='postgres', password=DB_PASSWORD, database='eicu')

print("=== eICU Cohort Re-extraction ===")

# 1. Build cohort from ds_eicu_shock
cur = conn.cursor()
cur.execute("""
SELECT ds.*, 
       p.age, p.gender, p.hospitalid, p.unittype,
       p.unitvisitnumber, p.admissionweight,
       p.unitdischargeoffset / 1440.0 AS los_days,
       p.unitdischargestatus
FROM public.ds_eicu_shock ds
JOIN eicu_crd.patient p ON ds.patientunitstayid = p.patientunitstayid
WHERE p.age NOT IN ('> 89', '')
  AND CAST(SPLIT_PART(p.age, ' ', 1) AS INTEGER) >= 18
ORDER BY ds.patientunitstayid
""")
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
print(f"Raw shock cohort: {len(rows)}")

df = pd.DataFrame(rows, columns=cols).reset_index(drop=True)
print(f"  SOFA: {df['sofa_total'].mean():.1f} (range {df['sofa_total'].min()}-{df['sofa_total'].max()})")
print(f"  Lactate: {df['lactate_max'].mean():.2f}")
print(f"  Mort: {df['mort_hosp'].mean()*100:.1f}%")

# 2. Exclusions compatible with MIMIC logic
# Check admission DX and diagnoses for exclusions
ids_list = df["patientunitstayid"].tolist()
ids_str = ",".join(str(x) for x in ids_list)

# Get all diagnoses for these patients
cur.execute(f"""
SELECT patientunitstayid, diagnosisstring 
FROM eicu_crd.diagnosis 
WHERE patientunitstayid IN ({ids_str})
""")
diag_rows = cur.fetchall()
print(f"\nDiagnoses retrieved: {len(diag_rows)}")

# Extract exclusion flags
exclude_set = set()
exclude_reasons = {}
for pid, dx in diag_rows:
    dx_lower = dx.lower()
    if 'cardiac arrest' in dx_lower:
        exclude_set.add(pid)
        exclude_reasons[pid] = exclude_reasons.get(pid, []) + ['cardiac_arrest']
    if 'cirrhosis' in dx_lower or 'liver cirrhosis' in dx_lower or 'hepatic cirrhosis' in dx_lower:
        exclude_set.add(pid)
        exclude_reasons[pid] = exclude_reasons.get(pid, []) + ['cirrhosis']
    if 'acute myocardial infarction' in dx_lower or 'ami' == dx_lower or 'st elevation mi' in dx_lower:
        exclude_set.add(pid)
        exclude_reasons[pid] = exclude_reasons.get(pid, []) + ['mi']
    if 'ecmo' in dx_lower or 'extracorporeal' in dx_lower or 'ecmo' == dx_lower.strip():
        exclude_set.add(pid)
        exclude_reasons[pid] = exclude_reasons.get(pid, []) + ['ecmo']

print(f"Patients with exclusions: {len(exclude_set)}")
print(f"  exclusions breakdown:")
from collections import Counter
reason_counts = Counter()
for reasons in exclude_reasons.values():
    for r in set(reasons):
        reason_counts[r] += 1
for k, v in reason_counts.most_common():
    print(f"    {k}: {v}")

# Apply exclusions
df_excl = df[~df["patientunitstayid"].isin(exclude_set)].copy()
print(f"\nAfter exclusions: {len(df_excl)}")

# 3. LOS >= 24h
df_excl["los_days"] = df_excl["los_days"].fillna(0)
df_excl = df_excl[df_excl["los_days"] >= 1].copy()
print(f"After LOS>=24h: {len(df_excl)}")

# 4. First ICU visit
df_excl = df_excl.sort_values("unitvisitnumber").groupby("patientunitstayid").first().reset_index()
print(f"After first visit: {len(df_excl)}")

# 5. Extract survival time (28-day capped hospital mortality)
# eICU has no 28-day follow-up, use hospital mortality
df_excl["surv_time"] = df_excl["los_days"].clip(upper=28)
df_excl["mort_28d"] = df_excl["mort_hosp"].astype(int)

print(f"\n=== Final Cohort ===")
print(f"N = {len(df_excl)}")
print(f"Mortality: {df_excl['mort_28d'].mean()*100:.1f}%")
print(f"SOFA: {df_excl['sofa_total'].mean():.1f} +/- {df_excl['sofa_total'].std():.1f}")
# Clean age (some are '> 89' even after filter)
def clean_age(v):
    try:
        return float(str(v).split()[0])
    except:
        return 90  # treat '> 89' as 90
df_excl['age_num'] = df_excl['age'].apply(clean_age)
print(f"Age: {df_excl['age_num'].mean():.1f}")

# 6. Extract DBP windows
print(f"\n=== Extracting DBP windows (0-48h, 8x6h) ===")
# Get chartoffset for each patient's DBP measurements
# pivoted_vital has heartrate, sysbp, diasbp, meanbp
ids_list2 = df_excl["patientunitstayid"].tolist()

# Process in chunks
dbp_records = []
for i in range(0, len(ids_list2), 500):
    chunk = ids_list2[i:i+500]
    ids_ph = ",".join(str(x) for x in chunk)
    try:
        cur.execute(f"""
        SELECT patientunitstayid, chartoffset, 
               COALESCE(ibp_diastolic, nibp_diastolic) AS diasbp
        FROM eicu_crd.pivoted_vital
        WHERE patientunitstayid IN ({ids_ph})
          AND COALESCE(ibp_diastolic, nibp_diastolic) IS NOT NULL
          AND chartoffset BETWEEN 0 AND 2880
        ORDER BY patientunitstayid, chartoffset
        """)
        dbp_records.extend(cur.fetchall())
    except Exception as e:
        print(f"  Chunk error: {e}")

print(f"DBP measurements: {len(dbp_records)}")

# Build 6-hour windows (0-48h)
dbp_df = pd.DataFrame(dbp_records, columns=["patientunitstayid", "chartoffset", "diasbp"])
dbp_df["window"] = dbp_df["chartoffset"] // 360  # 6-hour windows in minutes (360 min)

# Aggregate: mean DBP per window per patient
dbp_agg = dbp_df.groupby(["patientunitstayid", "window"])["diasbp"].mean().reset_index()
dbp_pivot = dbp_agg.pivot(index="patientunitstayid", columns="window", values="diasbp")
dbp_pivot.columns = [f"dbp_w{int(c)}" for c in dbp_pivot.columns]

# Ensure all 8 windows (0-7)
for w in range(8):
    col = f"dbp_w{w}"
    if col not in dbp_pivot.columns:
        dbp_pivot[col] = np.nan

# Reorder columns
dbp_pivot = dbp_pivot[[f"dbp_w{w}" for w in range(8)]]
dbp_pivot = dbp_pivot.reset_index()

print(f"DBP windows extracted: {len(dbp_pivot)} patients")

# 7. Extract vasopressor windows (binary: any VP in 6h window)
print(f"\n=== Extracting vasopressor windows ===")
# pivoted_treatment_vasopressor has vasopressor data
vp_records = []
for i in range(0, len(ids_list2), 500):
    chunk = ids_list2[i:i+500]
    ids_ph = ",".join(str(x) for x in chunk)
    try:
        cur.execute(f"""
        SELECT patientunitstayid, chartoffset, vasopressor
        FROM eicu_crd.pivoted_treatment_vasopressor
        WHERE patientunitstayid IN ({ids_ph})
          AND vasopressor = 1
          AND chartoffset BETWEEN 0 AND 2880
        ORDER BY patientunitstayid, chartoffset
        """)
        vp_records.extend(cur.fetchall())
    except Exception as e:
        if 'current transaction' not in str(e):
            print(f"  Chunk error: {e}")
        conn.rollback()

print(f"VP measurements: {len(vp_records)}")

vp_df = pd.DataFrame(vp_records, columns=["patientunitstayid", "chartoffset", "vasopressor"])
vp_df["window"] = vp_df["chartoffset"] // 360
vp_agg = vp_df.groupby(["patientunitstayid", "window"]).size().reset_index(name="count")
vp_agg["has_vp"] = 1  # binary: any vasopressor
vp_pivot = vp_agg.pivot(index="patientunitstayid", columns="window", values="has_vp").fillna(0).astype(int)
vp_pivot.columns = [f"vaso_w{int(c)}" for c in vp_pivot.columns]
for w in range(8):
    col = f"vaso_w{w}"
    if col not in vp_pivot.columns:
        vp_pivot[col] = 0
vp_pivot = vp_pivot[[f"vaso_w{w}" for w in range(8)]].reset_index()

print(f"VP windows extracted: {len(vp_pivot)} patients")

# 8. Merge all data
result = df_excl[["patientunitstayid", "sofa_total", "lactate_max", "mort_28d", "surv_time"]].copy()
result = result.merge(dbp_pivot, on="patientunitstayid", how="left")
result = result.merge(vp_pivot, on="patientunitstayid", how="left")

# Filter: >= 4 DBP windows
dbp_cols = [f"dbp_w{w}" for w in range(8)]
result["dbp_n"] = result[dbp_cols].notna().sum(axis=1)
before = len(result)
result = result[result["dbp_n"] >= 4].copy()
print(f"\nAfter >=4 DBP windows: {len(result)} / {before}")

# 9. Final summary
print(f"\n{'='*60}")
print(f"FINAL eICU COHORT (REAL SOFA)")
print(f"{'='*60}")
print(f"N = {len(result)}")
print(f"Mortality: {result['mort_28d'].mean()*100:.1f}%")
print(f"SOFA: {result['sofa_total'].mean():.1f} +/- {result['sofa_total'].std():.1f}")
print(f"DBP windows: mean={result['dbp_n'].mean():.1f}")

# 10. Save
result.to_csv("C:/Users/61656/.qclaw/workspace-agent-165f8164/dbp_vaspressor_trajectory/eicu_real_sofa_cohort.csv", index=False)
print(f"\nSaved: eicu_real_sofa_cohort.csv ({len(result)} rows, {len(result.columns)} cols)")

conn.close()
