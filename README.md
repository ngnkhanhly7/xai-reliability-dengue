# Reliability of LIME and Kernel SHAP for dengue severity classification

Code, configurations, saved results and figures for a study that evaluates LIME and Kernel SHAP explanations of a random forest on a public dengue severity cohort (798 DENV-2 patients, 138 severe; [Huang et al., 2020](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008960)).

Explanation reliability is split into five properties, each with its own check:

| Property | Check in this repo |
| --- | --- |
| Numerical correctness | Kernel SHAP vs. an independent exact Shapley enumeration (all 32 coalitions) |
| Local fidelity | LIME surrogate loss on its fitting sample and on 5,000 independent points |
| Stability | LIME seeds and budgets |
| Reference sensitivity | SHAP backgrounds of 20, 100 and 478 rows |
| Reliance correspondence | Synthetic marker arms C/R/B, with dependence measured by intervention |

## Stages

| Stage | Module (`python -m`) | Run ID | What it does |
| --- | --- | --- | --- |
| 1 | `src.run_experiments` | `dengue_20261006_v1` | Data audit, LR/RF baselines, exact SHAP check, LIME vs. SHAP on 20 test cases, stability, synthetic marker |
| 2 | `src.run_supplementary` | `dengue_20261006_supp_v2` | Kernel width, held-out evaluation, continuous-age LIME, within-bin age probe, background size |
| 3 | `src.mechanism_analysis` | `dengue_20261007_mechanism_v3` | Five matched surrogate controls and the within-pattern MSE decomposition |
| 4 | `src.replication_analysis` | `dengue_20261018_replication_v4` | Second model (gradient boosting), five further splits, distance of perturbed inputs to observed patients |

Stage 1 was locked before the test split was opened. Stages 2–4 are exploratory and were designed after Stage-1 results were seen; protocols and decision logs are in `report/`.

## Layout

- `src/` – pipeline code; `scripts/` – packaging and summary helpers
- `configs/` – frozen configuration per stage (includes the data SHA-256)
- `data/raw/dengue.csv` – the public CSV (also at `research/nguon/dengue_data_Final.csv`)
- `artifacts/<run_id>/` – split and case identifiers, environment, locks, saved models and LIME neighbourhoods
- `results/<run_id>/` – case-level attributions, diagnostics and verification files
- `figures/<run_id>/` – figures in PNG and PDF
- `report/` – protocols, decision logs and protocol amendments

## Setup (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Reproduce Stage 4 from the saved Stage 1–3 outputs (about 1 minute):

```powershell
.\.venv\Scripts\python.exe -m src.replication_analysis --run-id my_rerun --stage all
```

Results are written to `results/my_rerun/`. On split 42 the script asserts that it reproduces the saved Stage-2 perturbations and Stage-3 RMSE exactly.

## Notes

- If Git converts line endings (`core.autocrlf=true`), the CSV checksum differs from the configs although the content is the same; Stage 4 compares LF-normalized bytes.
- The re-verification steps of Stages 1–3 also check archived evidence bundles that are not included in this repository; their saved outputs are included in `results/` and `artifacts/`.
- All results describe explanations of fitted models on one retrospective cohort. They are not clinical or causal validation.
