"""
eICU full pipeline: cohort build → exclusions → DBP + vaso features → GMM transport
"""
import pg8000, pandas as pd, numpy as np, pickle, os, warnings

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")

warnings.filterwarnings("ignore")

BASE = DATA_DIR
conn = pg8000.connect(host='localhost', port=5433, user='postgres', password=DB_PASSWORD, database='eicu')
cur = conn.cursor()

# ==================== PHASE 1: COHORT ====================
print('=== Phase 1: Cohort ===')

# NE users
cur.execute("SELECT DISTINCT patientunitstayid FROM eicu_crd.infusiondrug WHERE drugname ILIKE '%norepinephrine%' AND drugrate IS NOT NULL AND TRIM(drugrate)~'^[0-9]+(\\.[0-9]+)?$'")
ne_set = set(r[0] for r in cur.fetchall())
print(f'  NE users: {len(ne_set)}')

# Septic shock diagnoses
cur.execute("SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis WHERE diagnosisoffset BETWEEN -1440 AND 1440 AND LOWER(diagnosisstring) LIKE '%septic shock%'")
shock_set = set(r[0] for r in cur.fetchall())

cur.execute("SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis WHERE diagnosisoffset BETWEEN -1440 AND 1440 AND diagnosisstring LIKE 'cardiovascular|shock / hypotension|sepsis%'")
sepsis_set = set(r[0] for r in cur.fetchall())

cand = (ne_set & shock_set) | (ne_set & sepsis_set)
print(f'  Candidates (NE + septic shock/sepsis): {len(cand)}')

# Exclusions via diagnosis
cur.execute("SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis WHERE diagnosisoffset BETWEEN -1440 AND 1440 AND LOWER(diagnosisstring) LIKE '%cardiac arrest%'")
ca_set = set(r[0] for r in cur.fetchall())

cur.execute("SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis WHERE diagnosisoffset BETWEEN -1440 AND 1440 AND LOWER(diagnosisstring) LIKE '%cirrhosis%' OR LOWER(diagnosisstring) LIKE '%liver cirrhosis%'")
cirr_set = set(r[0] for r in cur.fetchall())

cur.execute("SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis WHERE diagnosisoffset BETWEEN -1440 AND 1440 AND LOWER(diagnosisstring) LIKE '%acute myocardial infarction%'")
mi_set = set(r[0] for r in cur.fetchall())

cur.execute("SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis WHERE diagnosisoffset BETWEEN -1440 AND 1440 AND (LOWER(diagnosisstring) LIKE '%ecmo%' OR LOWER(diagnosisstring) LIKE '%extracorporeal%')")
ecmo_set = set(r[0] for r in cur.fetchall())

excl = ca_set | cirr_set | mi_set | ecmo_set
cand = cand - excl

ids = list(cand)
print(f'  After exclusions: {len(ids)}')

# ==================== PHASE 2: PATIENT DATA ====================
print('\n=== Phase 2: Patient data ===')
batch = 500
infos = {}
for i in range(0, len(ids), batch):
    b = ids[i:i+batch]; bh = ','.join(['%s']*len(b))
    cur.execute(f"""
        SELECT p.patientunitstayid, p.age, p.unitdischargeoffset, p.unitdischargestatus,
               p.unitvisitnumber, p.unitdischargeoffset
        FROM eicu_crd.patient p WHERE p.patientunitstayid IN ({bh})
    """, b)
    for r in cur.fetchall():
        infos[r[0]] = {'age': r[1], 'los_min': r[2], 'status': r[3], 'visit_num': r[4]}

# Build df
rows = []
for pid, info in infos.items():
    age = -1
    if info['age'] is not None:
        astr = str(info['age'])
        if astr.startswith('>'):
            age = 90
        else:
            try: age = float(astr)
            except: pass
    rows.append({'pid': pid, 'age': age, 'los_days': info['los_min']/1440 if info['los_min'] else 0,
                 'mort': 1 if info['status'] == 'Expired' else 0, 'visit': info['visit_num']})

df = pd.DataFrame(rows)
df = df[(df['age'] >= 18) & (df['los_days'] >= 1) & (df['visit'] == 1)]
print(f'  Age>=18, LOS>=24h, first visit: {len(df)}')

# eICU_sepsis_final SOFA + lactate
cur.execute("SELECT patientunitstayid, sofa_total, lactate_max, sofa_resp, sofa_coag, sofa_liver, sofa_cv, sofa_cns, sofa_renal FROM public.eicu_sepsis_final")
sofa_map = {r[0]: {'sofa': float(r[1]) if r[1] else None, 'lactate': float(r[2]) if r[2] else None} for r in cur.fetchall()}

df['sofa'] = df['pid'].map(lambda x: sofa_map.get(x, {}).get('sofa'))
df['lactate'] = df['pid'].map(lambda x: sofa_map.get(x, {}).get('lactate'))
print(f'  SOFA available: {df["sofa"].notna().sum()} ({df["sofa"].notna().mean()*100:.0f}%)')
print(f'  Lactate available: {df["lactate"].notna().sum()} ({df["lactate"].notna().mean()*100:.0f}%)')

# ==================== PHASE 3: DBP EXTRACTION ====================
print('\n=== Phase 3: DBP extraction ===')
pids = df['pid'].tolist()

all_dbp = []
for i in range(0, len(pids), 200):
    b = pids[i:i+200]; bh = ','.join(['%s']*len(b))
    # Arterial line DBP from vitalperiodic
    cur.execute(f"""
        SELECT v.patientunitstayid, v.systemicdiastolic::FLOAT AS dbp, v.observationoffset
        FROM eicu_crd.vitalperiodic v WHERE v.patientunitstayid IN ({bh})
          AND v.systemicdiastolic IS NOT NULL
    """, b)
    all_dbp.extend(cur.fetchall())
    # NIBP DBP from vitalaperiodic
    cur.execute(f"""
        SELECT v.patientunitstayid, v.noninvasivediastolic AS dbp, v.observationoffset
        FROM eicu_crd.vitalaperiodic v WHERE v.patientunitstayid IN ({bh})
          AND v.noninvasivediastolic IS NOT NULL
    """, b)
    all_dbp.extend(cur.fetchall())

dbp_df = pd.DataFrame(all_dbp, columns=['pid', 'dbp', 'offset'])
dbp_df['w'] = (dbp_df['offset'] / 360).astype(int)  # 6h windows (offset in min, 6h=360min)
dbp_df = dbp_df[(dbp_df['w'] >= 0) & (dbp_df['w'] <= 7)]

# Pivot DBP
dbp_piv = dbp_df.groupby(['pid', 'w'])['dbp'].mean().reset_index()
dbp_piv = dbp_piv.pivot_table(index='pid', columns='w', values='dbp').reset_index()
dbp_piv.columns = ['pid'] + [f'dbp_w{w}' for w in range(8)]
for w in range(8):
    if f'dbp_w{w}' not in dbp_piv.columns:
        dbp_piv[f'dbp_w{w}'] = np.nan

print(f'  DBP patients: {len(dbp_piv)}')

# ==================== PHASE 4: VASOPRESSOR EXTRACTION ====================
print('\n=== Phase 4: Vasopressor extraction (binary) ===')
all_vaso = []
for i in range(0, len(pids), 500):
    b = pids[i:i+500]; bh = ','.join(['%s']*len(b))
    cur.execute(f"""
        SELECT i.patientunitstayid, i.drugrate, i.drugname, i.infusionoffset
        FROM eicu_crd.infusiondrug i WHERE i.patientunitstayid IN ({bh})
          AND i.drugname ILIKE '%norepinephrine%'
          AND i.drugrate IS NOT NULL AND TRIM(i.drugrate) ~ '^[0-9]+(\\.[0-9]+)?\\s*$' 
    """, b)
    all_vaso.extend(cur.fetchall())

vaso_df = pd.DataFrame(all_vaso, columns=['pid', 'rate', 'drug', 'offset'])
# Handle missing offset (some records have null offset - use 0 as fallback)
vaso_df['offset'] = vaso_df['offset'].fillna(0).astype(int)
vaso_df['w'] = (vaso_df['offset'] / 360).astype(int)
vaso_df = vaso_df[(vaso_df['w'] >= 0) & (vaso_df['w'] <= 7)]

# Binary: any NE in window
vaso_set = vaso_df.groupby('pid')['w'].apply(set).reset_index()
for w in range(8):
    vaso_set[f'vaso_w{w}'] = vaso_set['w'].apply(lambda s: 1 if w in s else 0)
vaso_piv = vaso_set.drop('w', axis=1)

print(f'  Vaso patients: {len(vaso_piv)}')

# ==================== PHASE 5: MERGE ====================
print('\n=== Phase 5: Merge ===')
m = df.merge(dbp_piv, on='pid', how='inner').merge(vaso_piv, on='pid', how='inner')
m['n'] = m[[f'dbp_w{w}' for w in range(8)]].notna().sum(axis=1)
m = m[m['n'] >= 4].reset_index(drop=True)
print(f'  With >=4 DBP windows: {len(m)}')
print(f'  Mortality: {m["mort"].mean():.1%}')
print(f'  SOFA: {m["sofa"].mean():.1f}' if m['sofa'].notna().any() else '  SOFA: N/A')
print(f'  Age: {m["age"].mean():.1f}')

m.to_csv(f'{BASE}/eicu_diag_features.csv', index=False)
print(f'\nSaved: eicu_diag_features.csv ({len(m)} patients)')
conn.close()
