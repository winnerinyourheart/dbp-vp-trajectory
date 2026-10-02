-- MIMIC Hourly Time-Series (vitals + vasoactive agent exposure via norepinephrine_equivalent_dose table)
DROP TABLE IF EXISTS public.ds_mimic_hourly CASCADE;
CREATE TABLE public.ds_mimic_hourly (
    stay_id integer, hour_from_admit double precision, charttime timestamp,
    sbp double precision, dbp double precision, map double precision,
    hr double precision, spo2 double precision,
    ne_dose numeric
);
CREATE INDEX idx_mimic_hourly_stay ON public.ds_mimic_hourly(stay_id);
CREATE INDEX idx_mimic_hourly_time ON public.ds_mimic_hourly(stay_id, charttime);

INSERT INTO public.ds_mimic_hourly (stay_id, hour_from_admit, charttime, sbp, dbp, map, hr, spo2)
SELECT v.stay_id,
    EXTRACT(EPOCH FROM (v.charttime - c.intime)) / 3600.0,
    v.charttime,
    v.sbp, v.dbp, v.mbp, v.heart_rate, v.spo2
FROM mimiciv_derived.vitalsign v
INNER JOIN public.ds_mimic_shock c ON v.stay_id = c.stay_id
WHERE v.charttime BETWEEN c.intime AND c.intime + INTERVAL '72 hours'
    AND v.dbp IS NOT NULL;

UPDATE public.ds_mimic_hourly h
SET ne_dose = nd.norepinephrine_equivalent_dose
FROM mimiciv_derived.norepinephrine_equivalent_dose nd
WHERE h.stay_id = nd.stay_id
    AND h.charttime >= nd.starttime
    AND h.charttime < nd.endtime;

SELECT COUNT(*)::TEXT AS total_rows FROM public.ds_mimic_hourly
UNION ALL
SELECT COUNT(DISTINCT stay_id)::TEXT AS patients FROM public.ds_mimic_hourly
UNION ALL
SELECT ROUND(AVG(cnt)::numeric,1)::TEXT AS avg_hours
FROM (SELECT stay_id, COUNT(*) AS cnt FROM public.ds_mimic_hourly GROUP BY stay_id) t
UNION ALL
SELECT COUNT(*) FILTER (WHERE ne_dose IS NOT NULL)::TEXT AS rows_with_ne FROM public.ds_mimic_hourly;
