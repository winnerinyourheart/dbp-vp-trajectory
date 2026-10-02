# Joint DBP–Vasopressor Trajectory Analysis

Analysis code accompanying:

> **Joint Trajectories of Diastolic Blood Pressure Dynamics and Vasopressor Requirements Identify Prognostic Phenotypes in Septic Shock**
> *Clinical Epidemiology* — Manuscript ID 641615

---

## ⚠️ This repository contains code only — no data

MIMIC-IV and eICU-CRD are distributed under data use agreements that **prohibit redistribution of the data**. No patient-level data, derived feature tables, model objects or exported figures are included in this repository.

Access to both databases is obtained independently through [PhysioNet](https://physionet.org) after completing the required training and signing the data use agreements:

- MIMIC-IV v3.0 — DOI [10.13026/hxp0-hg59](https://doi.org/10.13026/hxp0-hg59)
- eICU-CRD v2.0 — DOI [10.13026/C2WM1R](https://doi.org/10.13026/C2WM1R)

---

## Requirements

- **Python 3.12**
- **PostgreSQL** with both databases loaded locally (default host `localhost:5433`)
- Python packages (`requirements.txt`):

```
numpy
pandas
scipy
scikit-learn
statsmodels
lifelines
pg8000
SQLAlchemy
matplotlib
Pillow
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Configuration

The scripts read two environment variables, so that no paths or credentials are hard-coded:

| Variable | Meaning | Default |
|---|---|---|
| `DBP_DATA_DIR` | Working directory holding the intermediate feature tables produced by steps 01–02 | directory of the script |
| `DBP_DB_PASSWORD` | Password for the local PostgreSQL `postgres` user | `postgres` |

```bash
export DBP_DATA_DIR=/path/to/working/directory
export DBP_DB_PASSWORD=your_password
```

The PostgreSQL connection settings (host `localhost`, port `5433`, user `postgres`, databases `mimiciv` and `eicu`) are stated at the top of each script and can be edited if your installation differs.

---

## Pipeline

Run the steps in order. Each step produces the inputs for the next.

| # | Script | Purpose | Key output |
|---|---|---|---|
| 01 | `01_cohort/build_cohort.sql` | MIMIC-IV septic shock cohort: Sepsis-3, vasopressor requirement within 48 h, lactate >2 mmol/L; exclusions for cardiac arrest, cirrhosis, acute MI, ECMO; first ICU stay, age ≥18, LOS ≥24 h | table `public.ds_mimic_shock` |
| 01 | `01_cohort/build_hourly_final.sql` | MIMIC-IV hourly time series with norepinephrine-equivalent dose | table `public.ds_mimic_hourly` |
| 01 | `01_cohort/02_build_mimic_features.py` | Eight 6-hour DBP windows and binary vasopressor windows; 28-day mortality and SOFA | `binary_mimic_data.csv` |
| 01 | `01_cohort/03_eicu_cohort.py` | eICU-CRD diagnostic cohort with equivalent criteria | `eicu_diag_features.csv` |
| 01 | `01_cohort/04_eicu_cohort_rebuild.py` | Rebuild / verify the eICU-CRD cohort | cohort tables |
| 01 | `01_cohort/05_eicu_sofa.py` | Derive the eICU-CRD total SOFA score | `eicu_real_sofa_cohort.csv` |
| 02 | `02_features/06_eicu_vasopressor_dose.py` | Parse eICU-CRD `infusiondrug` dose units, construct continuous dose and binary vasopressor windows | `eicu_diag_features_cont.csv` |
| 03 | `03_trajectory/07_primary_analysis.py` | **Primary analysis.** Gaussian mixture model (K = 6), class ordering, Cox models, bootstrap CIs, landmark analysis, missing-data / class-count / continuous-dose sensitivity analyses, feature-block decomposition, agent table | `rev_results_v1.json` |
| 03 | `03_trajectory/08_eicu_independent_gmm.py` | Independent eICU-CRD mixture model | eICU class assignments |
| 04 | `04_statistics/09_dbp_construct_decomposition.py` | DBP construct-by-construct increment against a treatment-adjusted reference model (Table 5, MIMIC-IV column) | `rev_dbp_role2_results.json` |
| 04 | `04_statistics/10_iptw_and_blocks.py` | Stabilised IPTW for vasopressin and feature-block decomposition | `rev_extra_results.json` |
| 04 | `04_statistics/11_eicu_planA_full.py` | **eICU-CRD full re-run** with the corrected unit conversion: classes, Cox models, bootstrap CIs, landmark analysis, DBP constructs, K = 5 sensitivity | `eicu_planA_v2_results.json`, `eicu_planA_v2_constructs.json` |
| 04 | `04_statistics/12_eicu_planA_extras.py` | Landmark confidence intervals, eICU-CRD hazard ratios, dose × DBP cross-classification | `eicu_planA_v2_extras.json` |
| 05 | `05_figures/13_figures_main.py` | Figures 1–4 and Supplementary Figures | `.png` / `.tif` |
| 05 | `05_figures/14_figures_final.py` | Final figure pass with legends moved out of the artwork | `.png` / `.tif` |
| 05 | `05_figures/15_figure4_replication.py` | Figure 4 (eICU-CRD replication) | `Figure_4.png` / `.tif` |
| 05 | `05_figures/16_figureS2_flow.py` | Supplementary Figure S2, cohort flow diagram | `Figure_S2.png` / `.tif` |
| 05 | `05_figures/17_figureS4_continuous_dose.py` | Supplementary Figure S4, continuous-dose sensitivity | `Figure_S4.png` / `.tif` |
| 05 | `05_figures/18_figureS5_missing_data.py` | Supplementary Figure S5, missing-data sensitivity | `Figure_S5.png` / `.tif` |

---

## Corrections applied in this revision

The re-audit reported in the Response to Reviewers identified and corrected four issues. The code in this repository is the **corrected** version.

1. **Missing-data handling.** The submitted analysis did not use last-observation-carried-forward as stated: MIMIC-IV applied `fillna(0)` to the 16-column feature matrix, and eICU-CRD applied within-column mean imputation. Missing DBP windows were therefore filled with 0 mmHg (a z-score of approximately −5.8). Both cohorts are now handled by LOCF (forward carry, with backward carry for leading missing values), applied identically.

2. **eICU-CRD vasopressor unit conversion** (`02_features/06_eicu_vasopressor_dose.py`). The `(mcg/min)` branch of the dose parser read `rate / 60 / weight`, treating a per-minute rate as an hourly one and deflating the dose 60-fold. Because a window was scored positive only above 0.01 µg/kg/min, 38.6% of the eICU-CRD cohort had a zero vasopressor dose in all eight windows; after correcting the branch to `rate / weight` this falls to 1.7%. The `(ml/hr)` branch retains its 32 µg/mL concentration assumption, which is disclosed in the Supplementary Methods. MIMIC-IV is unaffected: its exposure is read from the pre-computed `mimiciv_derived.norepinephrine_equivalent_dose` table.

3. **Cox penalisation convention.** Primary models are unpenalised; a ridge penaliser of 0.5 is applied only inside cross-validation folds. For the eICU-CRD DBP-construct analysis, where the number of evaluable windows is near-degenerate, a ridge penalty of 0.01 is applied identically to the reference model and to every augmented model so that likelihood-ratio statistics are comparable.

4. **Substring misclassification in the agent query.** The epinephrine pattern also matched *nor*epinephrine and phenyl*epinephrine*. The implemented query excludes both; the corrected estimates are reported in Table 6.

---

## Selected quantities for verification

| Quantity | MIMIC-IV | eICU-CRD |
|---|---|---|
| Analytic cohort, n | 4,869 | 3,624 |
| Class sizes (C0→C5) | 1365 / 307 / 1248 / 151 / 652 / 1146 | 638 / 1211 / 519 / 27 / 398 / 831 |
| Class mortality range | 2.1% – 27.7% | 6.4% – 23.1% |
| C-statistic, class + SOFA | 0.779 | 0.694 |
| Increment of class over SOFA (ΔC) | +0.040 | +0.021 |
| 48-hour landmark ΔC | +0.074 | +0.047 |
| DBP-dynamics block ΔC (likelihood-ratio χ², 5 df) | +0.018 (150.7) | +0.015 (128.8) |

---

## Notes on reproducibility

- Mixture models use `covariance_type="full"`, `random_state=42`, `n_init=20`, `max_iter=500`, `tol=1e-4`; classes are ordered by ascending mortality **after** fitting, and the mortality gradient plays no role in selecting K.
- Bootstrap confidence intervals use 300–400 resamples with fixed seeds, so point estimates and intervals reproduce exactly.
- DBP dynamics constructs are computed from **observed windows only** (no imputation), so that carry-forward cannot artificially flatten the trajectory.
- Continuous-dose analyses use a 6-hour area under the curve on the log scale, winsorised at the 99th percentile.

---

## Licence and citation

Released for academic reproducibility. If you use this code, please cite the article above.

The authors thank the MIT Laboratory for Computational Physiology, the MIMIC-IV and eICU-CRD teams, and the contributing centres for creating and maintaining the databases used in this study. The authors are solely responsible for the analysis, interpretation and conclusions presented here.
