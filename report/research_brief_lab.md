# Research brief: Reliability of LIME and Kernel SHAP for dengue severity classification

Prepared 7 October 2026 for a lab application focused on LIME and SHAP. This brief accompanies the [decision report](../bao-cao-quyet-dinh-phat-trien-lime-shap-dengue.md) and [full v1/v2 report](../bao-cao-tong-hop-lime-shap-dengue.md).

## Research question

When an explanation is numerically correct or stable across seeds, how well does it describe the fitted model, and which implementation choices limit that fidelity? The project investigates these questions separately using a small, reproducible medical classification setting. The public cohort contains 798 observations, including 138 severe cases, with age, sex, NS1, IgM and IgG as predictors. It is a retrospective DENV-2 cohort without released patient identifiers or observation timestamps. [Huang et al., 2020](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008960).

## Design and initial observations

A fixed stratified split produced 478 training, 160 validation and 160 test rows. The main model was a prespecified random forest; logistic regression was retained as a predictive baseline. The forest achieved test AUROC 0.756 and average precision 0.441, missing 20 of 28 severe cases at threshold 0.5. Explanation fidelity therefore concerns the behaviour of this imperfect fitted model.

Kernel SHAP was checked against independent enumeration of all 32 coalitions in the same empirical replacement game. The initial maximum attribution error was 3.33e-16. LIME's median fitting R² across 20 selected test cases was 0.282. Increasing perturbation budget reduced coefficient variation but did not improve top-three agreement consistently. Synthetic marker controls separately established model dependence before marker attributions were interpreted. These checks distinguish numerical correctness, fidelity, stability and reference sensitivity.

## Mechanism analysis

An exploratory follow-up reused ten selected validation cases and three sampling seeds. Five fixed controls used identical 2,000 raw fitting points, forest outputs and quartile kernel weights. Independent RNG evaluation pools contained 5,000 points each, but their v2 results had already been seen before the follow-up design.

Replacing only the age indicator with standardized continuous age reduced median held-out RMSE from **0.14862 to 0.10228**, a **31.2% reduction of the median**, with lower error in all ten selected cases. Removing the Ridge penalty had negligible effect at the tested width and budget. Adding ten pairwise terms reduced RMSE to 0.14490; a lookup model fitted to the 32 indicator patterns reached 0.14362.

A finite-sample weighted MSE identity separated within-pattern variation in forest outputs from error relative to evaluation-cell means. The former accounted for a median **93.1% of baseline MSE**. Because the four binary indicators identify their raw values, age coarsening is the information bottleneck in this representation. This is an empirical diagnosis in this pipeline, consistent with established analysis of tabular LIME, rather than a new explanation algorithm. [Garreau and von Luxburg, arXiv v3](https://arxiv.org/html/2008.11092v3).

Improved neighborhood RMSE did not guarantee better fidelity at the original instance: the continuous control improved point error in five cases and worsened it in five. It is therefore retained as a diagnostic control rather than selected as a universally superior explanation configuration.

## Contribution, limitations and evidence

The contribution is a reproducible evaluation that turns a low-fidelity observation into competing hypotheses and matched controls. The follow-up is exploratory, with ten raw profiles, eight quartile contexts and only three shared raw evaluation pools per domain. No new forest training or test predictions were used. It does not establish patient-level independence, external validity, causal effects or clinical usefulness. Background enlargement also does not resolve the definition of explanations under feature dependence. [Aas et al.](https://arxiv.org/abs/1903.10464).

All 949 files in the v2 evidence inventory remained unchanged. Surrogate checkpoints, raw neighborhoods, configuration locks, code, CSV/JSON outputs and [F8](../figures/dengue_20261007_mechanism_v3/F8_mechanism_controls.pdf) support inspection. Baseline coefficients and RMSE reproduced exactly; weighted OLS and the MSE identity passed independent numerical checks. [Verification](../results/dengue_20261007_mechanism_v3/verification.json).

For subsequent work with a clinical research team, the next questions are the prediction time and endpoint, patient/time separation, clinically credible references and perturbations, and evaluation with clinician feedback. These require an agreed clinical protocol and suitable data, rather than further tuning of the application dataset.
