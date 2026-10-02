"""
Extract continuous NE dose for diagnosis-based eICU cohort (3,624 patients)
Then apply dose threshold to redefine binary vaso_w for transport
"""
import pg8000, pandas as pd, numpy as np, os

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


BASE = DATA_DIR
e = pd.read_csv(f'{BASE}/eicu_diag_features.csv')
pids = e['pid'].tolist()
print(f'Cohort: {len(e)} patients')

conn = pg8000.connect(host='localhost', port=5433, user='postgres', password=DB_PASSWORD, database='eicu')
cur = conn.cursor()

# ===== GET PATIENT WEIGHTS =====
batch = 500
weights = {}
for i in range(0, len(pids), batch):
    b = pids[i:i+batch]; bh = ','.join(['%s']*len(b))
    cur.execute(f"SELECT patientunitstayid, admissionweight FROM eicu_crd.patient WHERE patientunitstayid IN ({bh})", b)
    for r in cur.fetchall():
        weights[r[0]] = float(r[1]) if r[1] else np.nan

e['weight'] = e['pid'].map(weights)

# ===== EXTRACT ALL NE DOSES =====
print('\nExtracting NE doses from infusiondrug...')
all_ne = []
for i in range(0, len(pids), 500):
    b = pids[i:i+500]; bh = ','.join(['%s']*len(b))
    cur.execute(f"""
        SELECT patientunitstayid, drugname, drugrate::FLOAT, infusionoffset
        FROM eicu_crd.infusiondrug
        WHERE patientunitstayid IN ({bh})
          AND drugname ILIKE '%norepinephrine%'
          AND drugrate IS NOT NULL AND TRIM(drugrate) ~ '^[0-9]+(\\.[0-9]+)?\\s*$'
    """, b)
    all_ne.extend(cur.fetchall())

print(f'  Raw NE records: {len(all_ne)}')
conn.close()

ne = pd.DataFrame(all_ne, columns=['pid', 'drug', 'rate', 'offset'])
ne['offset'] = ne['offset'].fillna(0).astype(int)

# ===== UNIT PARSING =====
def parse_ne_mcg_kg_min(row):
    """Convert NE drugrate to mcg/kg/min based on unit encoding in drugname"""
    drug = str(row['drug']).lower()
    rate = row['rate']
    
    if rate is None or rate <= 0:
        return 0.0
    
    w = row.get('_weight', 70)  # default 70kg if missing
    if w is None or w <= 0 or np.isnan(w):
        w = 70
    
    # Unit detection from drugname
    # NOTE (2026-10-02 correction): the '(mcg/min)' branch previously read
    # `rate / 60 / w`, which incorrectly treated mcg/min as mcg/h and divided
    # the dose by an extra factor of 60, collapsing genuine infusions below
    # the positivity threshold. mcg/min -> mcg/kg/min requires only /weight.
    if '(mcg/kg/min)' in drug:
        return rate  # already in mcg/kg/min
    elif '(mcg/min)' in drug:
        return rate / w  # mcg/min → mcg/kg/min
    elif '(ml/hr)' in drug or drug.endswith('()') or drug == 'norepinephrine':
        # ml/hr → standard concentration 32 mcg/ml
        mcg_per_min = rate * 32 / 60
        return mcg_per_min / w
    else:
        # Try to extract concentration from drugname
        import re
        m = re.search(r'\(?(\d+)\s*mg\s*/\s*(\d+)\s*ml\)?', drug)
        if m:
            mg = float(m.group(1)); ml = float(m.group(2))
            mcg_per_min = rate * (mg * 1000 / ml) / 60
            return mcg_per_min / w
        else:
            # Standard concentration fallback
            mcg_per_min = rate * 32 / 60
            return mcg_per_min / w

# Merge weight
ne = ne.merge(e[['pid', 'weight']], on='pid', how='left')
ne['_weight'] = ne['weight'].fillna(70)

# Convert to mcg/kg/min
ne['dose_mcg_kg_min'] = ne.apply(parse_ne_mcg_kg_min, axis=1)
ne['w'] = (ne['offset'] / 360).astype(int)
ne = ne[(ne['w'] >= 0) & (ne['w'] <= 7)]
ne['dose_mcg_kg_min'] = ne['dose_mcg_kg_min'].clip(0, 5)  # winsorize extreme

# ===== WINDOW AGGREGATION =====
# Mean dose per window
agg = ne.groupby(['pid', 'w'])['dose_mcg_kg_min'].mean().reset_index()
agg.columns = ['pid', 'w', 'mean_dose']

# Pivot
dose_piv = agg.pivot_table(index='pid', columns='w', values='mean_dose').reset_index()
dose_piv.columns = ['pid'] + [f'dose_w{w}' for w in range(8)]

# Merge back to eICU
e = e.merge(dose_piv, on='pid', how='left')

# Fill NaN windows with 0 (no NE in that window)
for w in range(8):
    col = f'dose_w{w}'
    if col not in e.columns:
        e[col] = 0.0
    else:
        e[col] = e[col].fillna(0.0)

# ===== TEST DIFFERENT THRESHOLDS =====
print('\n=== Threshold analysis (what vaso_w sparsity to target?) ===')
print(f'MIMIC approach: vaso_w=1 if any NE charted')
print(f'Current eICU: any NE record = 1 (nearly all windows)')

# MIMIC vaso rates from existing data
m = pd.read_csv(f'{BASE}/binary_mimic_data.csv')
mimic_vaso_pct = {}
for w in range(8):
    mimic_vaso_pct[w] = m[f'vaso_w{w}'].mean()
avg_mimic = np.mean(list(mimic_vaso_pct.values()))
print(f'\nMIMIC avg vaso% across windows: {avg_mimic:.1%}')
for w in range(8):
    print(f'  w{w}: {mimic_vaso_pct[w]:.1%}')

# Try thresholds
for thr in [0.01, 0.03, 0.05, 0.1, 0.2]:
    eicu_vaso_pcts = []
    for w in range(8):
        col = f'dose_w{w}'
        if col in e.columns:
            eicu_vaso_pcts.append((e[col] > thr).mean())
        else:
            eicu_vaso_pcts.append(0)
    avg_eicu = np.mean(eicu_vaso_pcts)
    print(f'\nThreshold >={thr:.2f} mcg/kg/min: avg vaso%={avg_eicu:.1%}')
    for w in range(8):
        bar = '#' * int(eicu_vaso_pcts[w] * 30)
        print(f'  w{w}: {eicu_vaso_pcts[w]:.1%} {bar}')

# Best threshold (match MIMIC avg ~ 8.6%)
best_thr = None
best_diff = 999
for thr in [t/100 for t in range(1, 51)]:
    eicu_pct = np.mean([(e[f'dose_w{w}'] > thr).mean() if f'dose_w{w}' in e.columns else 0 for w in range(8)])
    diff = abs(eicu_pct - avg_mimic)
    if diff < best_diff:
        best_diff = diff
        best_thr = thr

print(f'\nBest threshold to match MIMIC sparsity: {best_thr:.2f} mcg/kg/min')
print(f'  eICU vaso% at this threshold: {np.mean([(e[f"dose_w{w}"] > best_thr).mean() if f"dose_w{w}" in e.columns else 0 for w in range(8)]):.1%}')

# Save with this threshold
for w in range(8):
    col = f'dose_w{w}'
    e[f'vaso_w{w}'] = (e[col] > best_thr).astype(int)

# Also save continuous dose version
e.to_csv(f'{BASE}/eicu_diag_features_cont.csv', index=False)
print(f'\nSaved: eicu_diag_features_cont.csv')
