-- ============================================================
-- Build the MIMIC-IV septic shock cohort
--
-- Inclusion: Sepsis-3, a vasopressor requirement within 48 h of ICU
--            admission, and a serum lactate >2 mmol/L within 24 h.
-- Exclusion: cardiac arrest, cirrhosis, acute myocardial infarction,
--            extracorporeal membrane oxygenation.
-- Additional: first hospital stay, first ICU stay, age >=18 years,
--            ICU length of stay >=24 h.
--
-- Output: public.ds_mimic_shock — 4,883 patients.
--         The final analytic cohort is 4,869 after the additional
--         requirement of at least four evaluable DBP windows; that
--         step is applied in the analysis stage, not here.
--
-- The eICU-CRD replication cohort (n = 3,624) is built with
-- equivalent criteria by analysis/01_cohort/03_eicu_cohort.py.
-- ============================================================

-- ============================================================
-- MIMIC-IV COHORT
-- ============================================================
DROP TABLE IF EXISTS public.ds_mimic_shock CASCADE;

-- Stage 1-3: Sepsis-3 + vasopressor + lactate > 2 (first 24h)
CREATE TABLE public.ds_mimic_shock AS
WITH sepsis3 AS (
    SELECT s.subject_id, s.stay_id, icu.hadm_id, icu.intime, icu.outtime,
        s.suspected_infection_time,
        s.sofa_score AS sofa_delta
    FROM mimiciv_derived.sepsis3 s
    INNER JOIN mimiciv_icu.icustays icu ON s.stay_id = icu.stay_id
    WHERE s.sepsis3 = true AND s.suspected_infection_time IS NOT NULL
),
vaso AS (
    SELECT DISTINCT nd.stay_id
    FROM mimiciv_derived.norepinephrine_equivalent_dose nd
    INNER JOIN mimiciv_icu.icustays icu ON nd.stay_id = icu.stay_id
    WHERE nd.norepinephrine_equivalent_dose > 0
        AND nd.starttime >= icu.intime
        AND nd.starttime <= icu.intime + INTERVAL '48 hours'
),
lac AS (
    SELECT DISTINCT l.hadm_id
    FROM mimiciv_hosp.labevents l
    INNER JOIN sepsis3 s3 ON l.hadm_id = s3.hadm_id
    WHERE l.itemid IN (50813, 52442, 53183)
        AND l.valuenum IS NOT NULL AND l.valuenum > 2
        AND l.charttime BETWEEN s3.intime AND s3.intime + INTERVAL '24 hours'
),
excl_dx AS (
    SELECT DISTINCT hadm_id
    FROM mimiciv_hosp.diagnoses_icd
    WHERE (icd_version = 10 AND icd_code IN (
        'I469','I462','I468',            -- Cardiac arrest
        'K7030','K7031','K7291'          -- Cirrhosis
    )) OR (icd_version = 9 AND icd_code IN (
        '4275','42741','42742',          -- Cardiac arrest
        '5712','5715','5716','5723'      -- Cirrhosis
    ))
    UNION
    SELECT DISTINCT hadm_id
    FROM mimiciv_hosp.diagnoses_icd
    WHERE (icd_version = 10 AND icd_code LIKE 'I21%')   -- MI (acute)
       OR (icd_version = 9 AND icd_code LIKE '410%')    -- MI (acute)
),
excl_px AS (
    SELECT DISTINCT hadm_id
    FROM mimiciv_hosp.procedures_icd
    WHERE (icd_version = 10 AND icd_code IN (
        '5A15223','5A1522G','5A1522H','5A1522F'  -- ECMO
    )) OR (icd_version = 9 AND icd_code IN (
        '3965'                                      -- ECMO
    ))
),
excl AS (
    SELECT hadm_id FROM excl_dx
    UNION
    SELECT hadm_id FROM excl_px
)
SELECT DISTINCT s3.subject_id, s3.stay_id, s3.hadm_id,
    p.gender, p.anchor_age AS age, adm.race,
    s3.intime, s3.outtime,
    adm.hospital_expire_flag AS mort_hosp,
    adm.deathtime,
    -- 28-day mortality
    CASE WHEN adm.deathtime IS NOT NULL
              AND adm.deathtime <= s3.intime + INTERVAL '28 days'
         THEN 1 ELSE 0 END AS mort_28d,
    EXTRACT(EPOCH FROM (s3.outtime - s3.intime)) / 86400.0 AS los_icu,
    -- SOFA (first_day_sofa for absolute, sepsis3 for delta)
    fds.sofa AS sofa_total,
    fds.respiration AS sofa_resp,
    fds.coagulation AS sofa_coag,
    fds.liver AS sofa_liver,
    fds.cardiovascular AS sofa_cardio,
    fds.cns AS sofa_cns,
    fds.renal AS sofa_renal,
    -- Lactate
    ll.lactate_max,
    -- Vasopressor
    vd.max_vaso_dose,
    vd.first_vaso_time,
    -- Demographics
    EXTRACT(YEAR FROM s3.intime) - p.anchor_year AS age_at_admit,
    p.dod
FROM sepsis3 s3
INNER JOIN vaso v ON s3.stay_id = v.stay_id
INNER JOIN lac l ON s3.hadm_id = l.hadm_id
INNER JOIN mimiciv_derived.icustay_detail icu ON s3.stay_id = icu.stay_id
INNER JOIN mimiciv_hosp.patients p ON s3.subject_id = p.subject_id
INNER JOIN mimiciv_hosp.admissions adm ON s3.hadm_id = adm.hadm_id
LEFT JOIN mimiciv_derived.first_day_sofa fds ON s3.stay_id = fds.stay_id
LEFT JOIN (
    SELECT hadm_id, MAX(valuenum) AS lactate_max
    FROM mimiciv_hosp.labevents
    WHERE itemid IN (50813, 52442, 53183) AND valuenum IS NOT NULL
    GROUP BY hadm_id
) ll ON s3.hadm_id = ll.hadm_id
LEFT JOIN (
    SELECT nd.stay_id,
        MAX(nd.norepinephrine_equivalent_dose) AS max_vaso_dose,
        MIN(nd.starttime) AS first_vaso_time
    FROM mimiciv_derived.norepinephrine_equivalent_dose nd
    INNER JOIN mimiciv_icu.icustays icu ON nd.stay_id = icu.stay_id
    WHERE nd.norepinephrine_equivalent_dose > 0
        AND nd.starttime >= icu.intime
        AND nd.starttime <= icu.intime + INTERVAL '48 hours'
    GROUP BY nd.stay_id
) vd ON s3.stay_id = vd.stay_id
WHERE icu.first_hosp_stay = true
    AND icu.first_icu_stay = true
    AND icu.los_icu >= 1.0
    AND p.anchor_age >= 18
    AND s3.hadm_id NOT IN (SELECT hadm_id FROM excl);

-- Index
CREATE INDEX idx_mimic_shock_stay ON public.ds_mimic_shock(stay_id);
CREATE INDEX idx_mimic_shock_subject ON public.ds_mimic_shock(subject_id);

-- ============================================================
-- MIMIC HOURLY TIME-SERIES (first 72h)
-- ============================================================
DROP TABLE IF EXISTS public.ds_mimic_hourly CASCADE;

CREATE TABLE public.ds_mimic_hourly AS
SELECT v.stay_id,
    EXTRACT(EPOCH FROM (v.charttime - c.intime)) / 3600.0 AS hour_from_admit,
    v.charttime,
    v.sbp, v.dbp, v.mbp AS map,
    v.sbp_ni, v.dbp_ni, v.mbp_ni,
    v.heart_rate AS hr,
    v.spo2,
    nd.norepinephrine_equivalent_dose AS ne_dose,
    lac.lactate
FROM mimiciv_derived.vitalsign v
INNER JOIN public.ds_mimic_shock c ON v.stay_id = c.stay_id
LEFT JOIN LATERAL (
    SELECT norepinephrine_equivalent_dose
    FROM mimiciv_derived.norepinephrine_equivalent_dose nd2
    WHERE nd2.stay_id = v.stay_id
        AND nd2.starttime <= v.charttime
        AND nd2.endtime > v.charttime
    ORDER BY nd2.starttime DESC
    LIMIT 1
) nd ON true
LEFT JOIN LATERAL (
    SELECT l.valuenum AS lactate
    FROM mimiciv_hosp.labevents l
    WHERE l.hadm_id = c.hadm_id
        AND l.itemid IN (50813, 52442, 53183)
        AND l.valuenum IS NOT NULL
        AND l.charttime <= v.charttime
        AND l.charttime > v.charttime - INTERVAL '4 hours'
    ORDER BY l.charttime DESC
    LIMIT 1
) lac ON true
WHERE v.charttime BETWEEN c.intime AND c.intime + INTERVAL '72 hours'
    AND v.dbp IS NOT NULL;

-- Index
CREATE INDEX idx_mimic_hourly_stay ON public.ds_mimic_hourly(stay_id);

-- ============================================================
-- Summary
-- ============================================================
SELECT '=== MIMIC-IV ===' AS db, COUNT(*)::TEXT, 
       ROUND(AVG(sofa_total)::numeric,1),
       ROUND(100.0*SUM(mort_28d)/COUNT(*),1),
       ROUND(AVG(age)::numeric,1)
FROM public.ds_mimic_shock;
