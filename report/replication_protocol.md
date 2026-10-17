# Stage 4 protocol (exploratory): second model, repeated splits, perturbation plausibility

Written 18 October 2026, before any Stage-4 computation. Stage 4 is post hoc: Stage 1–3 results are known. It answers reviewer feedback on generalization and clinical plausibility. It does not change any Stage 1–3 model, split, case set, threshold or reported number.

## Questions

1. **Replication across fitted models and splits.** Do the two main findings keep their direction (a) on a second nonlinear model and (b) on other train/validation/test splits?
   - F1 (LIME representation bottleneck): with identical LIME fit points, model outputs and quartile kernel weights, replacing the quartile age indicator by standardized age lowers held-out weighted RMSE; within-pattern model variation is the majority of quartile-surrogate MSE.
   - F2 (SHAP reference sensitivity): exact Shapley attributions move with a 20-row background; a nested 100-row background moves closer to the full-training-set explanation.
2. **Plausibility audit.** How far are LIME perturbations and SHAP hybrid rows from observed patients?

## Fixed design

- Data: same CSV (SHA-256 checked). Same split procedure as Stage 1 (stratified 20% test, then 25% of remainder validation) with split seeds 42 (original) and 2027–2031.
- Models, untuned and fixed before running:
  - RF: Stage-1 hyperparameters (200 trees, max_depth 5, min_samples_leaf 3, max_features sqrt, seed 11). On split 42 it must reproduce the saved Stage-1 RF predictions exactly.
  - GBM: `GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=3, random_state=11)`.
- Cases: per split, 10 validation cases, 5 per label, seed 1101 (Stage-2 rule). On split 42 they must equal the Stage-2 IDs.
- LIME: Stage-2 sampler (quartile, w0 = 0.75·sqrt(5), 2,000 fit points, fit seeds 301–303, 5,000 evaluation points with seed + 1000, forced original removed). Two arms fitted on identical points, outputs and quartile weights: quartile ridge and continuous-age ridge (alpha 1). Held-out RMSE uses common quartile weights. Within-pattern fraction uses the Stage-3 identity. On split 42 with RF, fit points must equal the saved Stage-2 neighbourhoods.
- SHAP: exact enumeration of the empirical marginal game (equal to full-enumeration Kernel SHAP, verified in Stages 1–2), backgrounds n20 seeds 201–205, nested n100, full training set (Stage-2 rule).
- Test metrics (AUROC, AP, sensitivity at 0.5) are reported as context. They are new test queries for GBM and for all non-42 splits.

## Decision rule (stated before running)

Report direction counts across the 12 model × split configurations: number of configurations where the continuous arm lowers median held-out RMSE; median within-pattern fraction; number where the 100-row background lowers baseline SD and L1 distance to full train. No p-values; no configuration is selected as best.

## Plausibility audit

On split 42 / RF Stage-2 inputs (10 validation cases, seeds 301–303): quartile and Gaussian LIME fit points, and SHAP hybrid rows for the 30 non-trivial coalitions under n20, n100 and full backgrounds. For each synthetic row: age outside the training range; non-integer age; distance to the nearest training patient, d = |Δage| / SD(age, train) + Hamming distance on sex/NS1/IgM/IgG; and whether no training patient with the same sex/laboratory pattern is within 5 years of age. Reference: the same quantities for the 160 real validation patients. No model predictions are needed.
