"""Descriptive, post-result summary for the lab-application decision report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = "dengue_20261007_mechanism_v3"
directory = ROOT / "results" / RUN
fits = pd.read_csv(directory / "fits.csv")
cases = pd.read_csv(directory / "per_case.csv")
decomposition = pd.read_csv(directory / "decomposition_per_case.csv")
frame = pd.read_csv(ROOT / "data/raw/dengue.csv")
selected = frame.iloc[sorted(fits.row_id.unique())].copy()
selected.insert(0, "row_id", selected.index)
selected["age_bin"] = np.digitize(selected.age, [36, 59.5, 74], right=True)
context = ["age_bin", "sex", "NS1", "IgM", "IgG"]
selected[["row_id", "Severity", "age", "sex", "NS1", "IgM", "IgG", "age_bin"]].to_csv(directory / "selected_contexts.csv", index=False)
summary = pd.read_csv(directory / "summary.csv").set_index("arm")
reference = cases[cases.arm == "quartile_ridge"].set_index("row_id")
continuous = cases[cases.arm == "continuous_same_samples"].set_index("row_id")
floor = decomposition[decomposition.arm == "quartile_ridge"]
matched = cases.merge(selected[["row_id"] + context], on="row_id")
# Exploratory sensitivity: equal weight to each of eight coarse contexts.
balanced = matched.groupby(["arm"] + context)[["quartile_rmse", "point_error"]].median().groupby("arm").median()
balanced.to_csv(directory / "context_balanced_sensitivity.csv")
point_comparison = reference[["point_error"]].join(continuous[["point_error"]], lsuffix="_quartile", rsuffix="_continuous")
point_comparison.to_csv(directory / "paired_point_errors.csv")
result = {
    "analysis_date_client": "2026-10-07",
    "status": "descriptive_summary_after_v3_results",
    "n_selected_cases": len(selected),
    "unique_full_input_profiles": len(selected[["age", "sex", "NS1", "IgM", "IgG"]].drop_duplicates()),
    "unique_quartile_contexts": len(selected[context].drop_duplicates()),
    "unique_quartile_raw_evaluation_pools": fits.quartile_raw_sha256.nunique(),
    "unique_gaussian_raw_evaluation_pools": fits.gaussian_raw_sha256.nunique(),
    "median_rmse_fraction_reduction_continuous": 1 - summary.loc["continuous_same_samples", "quartile_rmse"] / summary.loc["quartile_ridge", "quartile_rmse"],
    "baseline_within_cell_mse_fraction_median": floor.within_cell_fraction_of_mse.median(),
    "baseline_within_cell_mse_fraction_min": floor.within_cell_fraction_of_mse.min(),
    "baseline_within_cell_mse_fraction_max": floor.within_cell_fraction_of_mse.max(),
    "empirical_floor_rmse_median": floor.empirical_floor_rmse.median(),
    "minimum_fit_cell_count": int(fits.minimum_observed_cell_count.min()),
    "context_balanced_sensitivity": balanced.reset_index().to_dict(orient="records"),
    "point_error_pairs": point_comparison.reset_index().to_dict(orient="records"),
    "gaussian_outside_age_fraction_median": fits.groupby(["row_id", "fit_seed"]).gaussian_outside_age_fraction.first().median(),
    "summary": summary.reset_index().to_dict(orient="records"),
}
(directory / "decision_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key not in {"summary", "point_error_pairs"}}, ensure_ascii=False, indent=2))
