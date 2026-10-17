"""Fixed, exploratory v3 controls; reads v1/v2 without modifying their files."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import sys
import time
import warnings
import zipfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits
import yaml

from .common import FEATURES, ROOT, now, read_json, sha, write_json
from .supplementary_lime import array_hash, encode, make_explainer, score, weights


def load_arrays(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def cell_ids(z):
    assert np.all((z == 0) | (z == 1))
    return (z.astype(int) * (2 ** np.arange(5))).sum(axis=1)


def cell_means(z, y, w):
    ids = cell_ids(z)
    mass = np.bincount(ids, weights=w, minlength=32)
    totals = np.bincount(ids, weights=w * y, minlength=32)
    means = np.full(32, np.average(y, weights=w))
    np.divide(totals, mass, out=means, where=mass > 0)
    return means, mass, np.bincount(ids, minlength=32)


def pairwise(z):
    return np.column_stack([z] + [z[:, i] * z[:, j] for i, j in combinations(range(5), 2)])


def design(arm, qz, cz):
    if arm == "continuous_same_samples":
        return cz
    if arm == "quartile_pairwise":
        result = pairwise(qz)
        expected = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(qz)
        assert np.array_equal(result, expected)
        return result
    return qz


def fit_arm(arm, qz, cz, y, w, alpha, seed):
    matrix = design(arm, qz, cz)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if arm == "quartile_cell_mean":
            means, mass, counts = cell_means(qz, y, w)
            fitted = {"kind": "cell_mean", "means": means, "mass": mass, "counts": counts}
        elif arm == "quartile_ols":
            weighted_x = np.column_stack([np.ones(len(matrix)), matrix]) * np.sqrt(w[:, None])
            beta, _, rank, _ = np.linalg.lstsq(weighted_x, y * np.sqrt(w), rcond=None)
            reference = LinearRegression().fit(matrix, y, sample_weight=w)
            check = np.max(abs(weighted_x @ beta / np.sqrt(w) - reference.predict(matrix)))
            assert check < 1e-10 and rank == 6
            fitted = {"kind": "ols", "intercept": beta[0], "coef": beta[1:], "rank": int(rank),
                      "sklearn_prediction_max_error": float(check)}
        else:
            regression = Ridge(alpha=alpha, fit_intercept=True, random_state=seed)
            regression.fit(matrix, y, sample_weight=w)
            fitted = {"kind": "ridge", "regressor": regression}
    fitted.update({"arm": arm, "fit_seconds": time.perf_counter() - started,
                   "warnings": [str(item.message) for item in caught]})
    return fitted


def predict(fitted, qz, cz):
    matrix = design(fitted["arm"], qz, cz)
    if fitted["kind"] == "cell_mean":
        return fitted["means"][cell_ids(qz)]
    if fitted["kind"] == "ols":
        return fitted["intercept"] + matrix @ fitted["coef"]
    return fitted["regressor"].predict(matrix)


def initialize(config_path, run_id=None):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if run_id:
        cfg["run_id"] = run_id
    out = {name: ROOT / name / cfg["run_id"] for name in ["artifacts", "results", "figures"]}
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "release/dengue_evidence_v2.zip") as archive:
        inventory = json.loads(archive.read("EVIDENCE_MANIFEST.json"))["sha256"]
    mismatches = [path for path, digest in inventory.items() if sha(ROOT / path) != digest]
    assert not mismatches, f"Current v2 evidence differs from archive: {mismatches}"
    assert sha(ROOT / cfg["data_path"]) == cfg["data_sha256"]
    baseline = ROOT / "artifacts" / cfg["baseline_run"]
    parent = ROOT / "artifacts" / cfg["parent_run"]
    splits = read_json(baseline / "splits/split_ids.json")
    assert set(cfg["validation_ids"]) <= set(splits["validation"])
    assert not (set(cfg["validation_ids"]) & set(splits["test"]))
    assert cfg["validation_ids"] == read_json(parent / "splits/validation_ids.json")["row_ids"]
    inputs = [config_path, ROOT / cfg["protocol_file"], Path(__file__), ROOT / "src/supplementary_lime.py",
              ROOT / "src/common.py", baseline / "models/model_rf_seed11.joblib",
              baseline / "splits/split_ids.json", ROOT / cfg["data_path"],
              ROOT / "results" / cfg["parent_run"] / "lime_ablation.csv"]
    for row in cfg["validation_ids"]:
        for seed in cfg["fit_seeds"]:
            for representation in ["quartile", "continuous"]:
                inputs += [parent / "neighborhoods" / f"{representation}_row{row}_seed{seed}.{ext}" for ext in ["npz", "json"]]
            inputs += [parent / "surrogates" / f"quartile_w1_row{row}_seed{seed}.joblib"]
    input_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in inputs}
    lock_path = out["artifacts"] / "locked_before_v3_fits.json"
    if lock_path.exists():
        lock = read_json(lock_path)
        assert lock["input_sha256"] == input_hashes, "Frozen v3 inputs changed: use a new run_id."
    else:
        lock = {"run_id": cfg["run_id"], "analysis_date_client": cfg["analysis_date_client"],
                "locked_at_machine_clock": now(), "exploratory_after_v1_v2": True,
                "evaluation_pools_previously_seen": True, "validation_ids": cfg["validation_ids"],
                "config": cfg, "input_sha256": input_hashes, "protected_v2_sha256": inventory,
                "archive_v2_sha256": sha(ROOT / "release/dengue_evidence_v2.zip"),
                "new_model_training": False, "new_test_predictions": False}
        write_json(lock_path, lock)
        write_json(out["artifacts"] / "environment.json", {
            "python": sys.version, "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in
                         ["numpy", "pandas", "scipy", "scikit-learn", "lime", "shap", "matplotlib", "joblib"]},
            "threads": 1})
    frame = pd.read_csv(ROOT / cfg["data_path"])
    return cfg, out, lock, frame[FEATURES].to_numpy(float), splits


def execute(cfg, out, X, splits):
    parent = ROOT / "artifacts" / cfg["parent_run"]
    w0 = cfg["width_multiplier"] * np.sqrt(5)
    training = X[splits["train"]]
    records, decompositions = [], []
    model_dir = out["artifacts"] / "surrogates"
    model_dir.mkdir(exist_ok=True)
    for row in cfg["validation_ids"]:
        for seed in cfg["fit_seeds"]:
            x = X[row]
            q = load_arrays(parent / "neighborhoods" / f"quartile_row{row}_seed{seed}.npz")
            gaussian = load_arrays(parent / "neighborhoods" / f"continuous_row{row}_seed{seed}.npz")
            assert len(q["fit_y"]) == cfg["fit_budget"] and len(q["eval_y"]) == cfg["eval_budget"]
            assert array_hash(q["fit_raw"]) != array_hash(q["eval_raw"])
            assert np.array_equal(q["fit_raw"][0], x)
            exq = make_explainer(training, seed, "quartile", w0)
            exc = make_explainer(training, seed, "continuous", w0)
            fit_q, fit_c = encode(exq, q["fit_raw"], x), encode(exc, q["fit_raw"], x)
            assert np.array_equal(fit_q, q["fit_z"])
            fit_w = weights(q["fit_distances"], w0)
            qx, cx = encode(exq, x[None, :], x), encode(exc, x[None, :], x)
            eval_means, eval_mass, _ = cell_means(q["eval_z"], q["eval_y"], weights(q["eval_distances"], w0))
            qeval_w = weights(q["eval_distances"], w0)
            floor_mse = float(np.average((q["eval_y"] - eval_means[cell_ids(q["eval_z"])] ) ** 2, weights=qeval_w))
            for arm in cfg["arms"]:
                fitted = fit_arm(arm, fit_q, fit_c, q["fit_y"], fit_w, cfg["ridge_alpha"], seed)
                record = {"row_id": row, "fit_seed": seed, "evaluation_seed": seed + 1000, "arm": arm,
                          "split": "validation", "fit_raw_sha256": array_hash(q["fit_raw"]),
                          "fit_weights_sha256": array_hash(fit_w), "fit_y_sha256": array_hash(q["fit_y"]),
                          "point_prediction": float(predict(fitted, qx, cx)[0]), "rf_probability": float(q["fit_y"][0]),
                          "fit_seconds": fitted["fit_seconds"], "warnings": " | ".join(fitted["warnings"]),
                          "n_terms": 32 if arm == "quartile_cell_mean" else design(arm, fit_q, fit_c).shape[1]}
                record["point_error"] = abs(record["point_prediction"] - record["rf_probability"])
                for name, pool in [("fit", q), ("quartile", q), ("gaussian", gaussian)]:
                    subset = "fit" if name == "fit" else "eval"
                    raw, y = pool[subset + "_raw"], pool[subset + "_y"]
                    qz, cz = encode(exq, raw, x), encode(exc, raw, x)
                    w = fit_w if name == "fit" else weights(pool["eval_distances"], w0)
                    prediction = predict(fitted, qz, cz)
                    record.update({name + "_" + key: value for key, value in score(y, prediction, w).items()})
                    record[name + "_outside_probability_fraction"] = float(np.mean((prediction < 0) | (prediction > 1)))
                    if name != "fit":
                        record[name + "_raw_sha256"] = array_hash(raw)
                        record[name + "_weights_sha256"] = array_hash(w)
                    if name == "gaussian":
                        valid = (raw[:, 0] >= training[:, 0].min()) & (raw[:, 0] <= training[:, 0].max())
                        record["gaussian_outside_age_fraction"] = float(np.mean(~valid))
                        record["gaussian_in_range_rmse"] = score(y[valid], prediction[valid], w[valid])["rmse"]
                    if name == "quartile" and arm != "continuous_same_samples":
                        all_patterns = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)
                        cell_prediction = predict(fitted, all_patterns, all_patterns)
                        between = float(np.sum(eval_mass * (eval_means - cell_prediction)**2) / eval_mass.sum())
                        mse = record["quartile_rmse"] ** 2
                        identity_error = abs(mse - floor_mse - between)
                        assert identity_error < 1e-12
                        decompositions.append({"row_id": row, "fit_seed": seed, "arm": arm, "mse": mse,
                                               "within_cell_mse": floor_mse, "between_cell_mse": between,
                                               "within_cell_fraction_of_mse": floor_mse / mse,
                                               "empirical_floor_rmse": np.sqrt(floor_mse), "identity_error": identity_error})
                if arm == "quartile_ridge":
                    reference = joblib.load(parent / "surrogates" / f"quartile_w1_row{row}_seed{seed}.joblib")
                    record["saved_v2_coefficient_max_error"] = float(np.max(abs(reference.coef_ - fitted["regressor"].coef_)))
                    record["saved_v2_intercept_error"] = float(abs(reference.intercept_ - fitted["regressor"].intercept_))
                    assert record["saved_v2_coefficient_max_error"] < 1e-10 and record["saved_v2_intercept_error"] < 1e-10
                if arm == "quartile_ols":
                    record["ols_sklearn_max_error"] = fitted["sklearn_prediction_max_error"]
                if arm == "quartile_cell_mean":
                    counts = fitted["counts"]
                    unseen = counts[cell_ids(q["eval_z"])] == 0
                    record.update({"fit_cells_observed": int(np.count_nonzero(counts)), "minimum_observed_cell_count": int(counts[counts > 0].min()),
                                   "unseen_eval_fraction": float(unseen.mean()), "unseen_eval_weight_fraction": float(qeval_w[unseen].sum() / qeval_w.sum())})
                fitted.update({"age_mean": float(exc.scaler.mean_[0]), "age_scale": float(exc.scaler.scale_[0]),
                               "age_bins": [36., 59.5, 74.], "original_raw": x, "shared_fit_weight_encoding": "quartile"})
                joblib.dump(fitted, model_dir / f"{arm}_row{row}_seed{seed}.joblib")
                records.append(record)
        print(f"v3 validation row {row}: five fixed arms complete", flush=True)
    table = pd.DataFrame(records)
    decomp = pd.DataFrame(decompositions)
    table.to_csv(out["results"] / "fits.csv", index=False)
    decomp.to_csv(out["results"] / "mse_decomposition.csv", index=False)
    metrics = ["point_error", "fit_rmse", "fit_r2", "quartile_rmse", "quartile_r2", "gaussian_rmse", "gaussian_r2", "gaussian_in_range_rmse"]
    per_case = table.groupby(["arm", "row_id"], sort=False)[metrics].median().reset_index()
    per_case.to_csv(out["results"] / "per_case.csv", index=False)
    summary = per_case.groupby("arm", sort=False)[metrics].median().reset_index()
    summary.to_csv(out["results"] / "summary.csv", index=False)
    paired = []
    baseline = per_case[per_case.arm == "quartile_ridge"].set_index("row_id")
    for arm in cfg["arms"][1:]:
        other = per_case[per_case.arm == arm].set_index("row_id")
        for metric in metrics:
            delta = other[metric] - baseline[metric]
            paired.append({"arm": arm, "metric": metric, "n_cases": len(delta), "median_paired_delta": delta.median(),
                           "cases_lower": int((delta < -1e-12).sum()), "cases_higher": int((delta > 1e-12).sum()),
                           "cases_equal": int((abs(delta) <= 1e-12).sum())})
    pd.DataFrame(paired).to_csv(out["results"] / "paired_effects.csv", index=False)
    decomp_case = decomp.groupby(["arm", "row_id"], sort=False).median(numeric_only=True).reset_index()
    decomp_case.to_csv(out["results"] / "decomposition_per_case.csv", index=False)
    decomp_case.groupby("arm", sort=False).median(numeric_only=True).reset_index().to_csv(out["results"] / "decomposition_summary.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    labels = ["Q Ridge", "Q OLS", "Continuous\nsame samples", "Q pairwise", "Q cell mean"]
    for _, group in per_case.groupby("row_id", sort=False):
        values = group.set_index("arm").loc[cfg["arms"], "quartile_rmse"].to_numpy()
        axes[0].plot(range(5), values, color="#aab2bd", linewidth=.8, alpha=.65)
    medians = summary.set_index("arm").loc[cfg["arms"], "quartile_rmse"].to_numpy()
    axes[0].plot(range(5), medians, "o-", color="#176b87", linewidth=2)
    axes[0].set(xticks=range(5), xticklabels=labels, ylabel="Held-out weighted RMSE", title="Identical fit points, RF targets and weights")
    chosen = decomp_case[decomp_case.arm == "quartile_ridge"].sort_values("row_id")
    positions = np.arange(len(chosen))
    axes[1].bar(positions, chosen.within_cell_fraction_of_mse, label="Within-pattern variance", color="#176b87")
    axes[1].bar(positions, 1 - chosen.within_cell_fraction_of_mse, bottom=chosen.within_cell_fraction_of_mse,
                label="Between-pattern error", color="#e3a65d")
    axes[1].set(xticks=positions, xticklabels=chosen.row_id.astype(str), ylim=(0, 1), xlabel="Validation row ID", ylabel="Fraction of baseline MSE", title="Finite evaluation-sample decomposition")
    axes[1].legend(fontsize=8, loc="lower right")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    for extension in ["png", "pdf"]:
        figure.savefig(out["figures"] / f"F8_mechanism_controls.{extension}", dpi=200)
    plt.close(figure)
    return table, decomp


def verify(cfg, out, lock, X, splits):
    table = pd.read_csv(out["results"] / "fits.csv")
    decomp = pd.read_csv(out["results"] / "mse_decomposition.csv")
    assert len(table) == 150 and len(decomp) == 120
    assert set(table.row_id) == set(cfg["validation_ids"]) and set(table.arm) == set(cfg["arms"])
    for _, group in table.groupby(["row_id", "fit_seed"]):
        assert len(group) == 5
        for column in ["fit_raw_sha256", "fit_y_sha256", "fit_weights_sha256", "quartile_raw_sha256", "quartile_weights_sha256", "gaussian_raw_sha256", "gaussian_weights_sha256"]:
            assert group[column].nunique() == 1
    old = pd.read_csv(ROOT / "results" / cfg["parent_run"] / "lime_ablation.csv")
    reference = old[(old.representation == "quartile") & (old.width_factor == 1)]
    baseline = table[table.arm == "quartile_ridge"]
    joined = baseline.merge(reference, on=["row_id", "fit_seed"], suffixes=("_v3", "_v2"))
    rmse_error = float(abs(joined.quartile_rmse - joined.heldout_common_quartile_rmse).max())
    assert len(joined) == 30 and rmse_error < 1e-10
    row, seed = cfg["validation_ids"][0], cfg["fit_seeds"][0]
    q = load_arrays(ROOT / "artifacts" / cfg["parent_run"] / "neighborhoods" / f"quartile_row{row}_seed{seed}.npz")
    w0 = cfg["width_multiplier"] * np.sqrt(5)
    exc = make_explainer(X[splits["train"]], seed, "continuous", w0)
    cz = encode(exc, q["fit_raw"], X[row])
    eval_cz = encode(exc, q["eval_raw"], X[row])
    rerun_errors = {}
    for arm in cfg["arms"]:
        fresh = fit_arm(arm, q["fit_z"], cz, q["fit_y"], weights(q["fit_distances"], w0), cfg["ridge_alpha"], seed)
        saved = joblib.load(out["artifacts"] / "surrogates" / f"{arm}_row{row}_seed{seed}.joblib")
        error = float(np.max(abs(predict(fresh, q["eval_z"], eval_cz) - predict(saved, q["eval_z"], eval_cz))))
        assert error < 1e-10
        rerun_errors[arm] = error
    changed = [path for path, digest in lock["protected_v2_sha256"].items() if sha(ROOT / path) != digest]
    assert not changed
    result = {"passed": True, "verified_at_machine_clock": now(), "analysis_date_client": cfg["analysis_date_client"],
              "fits": len(table), "validation_cases": len(cfg["validation_ids"]), "test_queries": 0, "rf_refits": 0,
              "previous_files_unchanged": len(lock["protected_v2_sha256"]), "shared_fit_points_targets_weights": True,
              "common_evaluation_points_weights": True, "evaluation_pools_previously_seen": True,
              "baseline_coefficient_max_error": float(baseline.saved_v2_coefficient_max_error.max()),
              "baseline_heldout_rmse_max_error": rmse_error, "ols_sklearn_prediction_max_error": float(table.ols_sklearn_max_error.max()),
              "mse_identity_max_error": float(decomp.identity_error.max()), "fresh_refit_prediction_errors": rerun_errors,
              "cell_mean_min_cells": int(table.fit_cells_observed.min()),
              "cell_mean_max_unseen_eval_fraction": float(table.unseen_eval_fraction.max()),
              "warnings_count": int(table.warnings.notna().sum())}
    write_json(out["results"] / "verification.json", result)
    outputs = {path.relative_to(ROOT).as_posix(): sha(path) for directory in out.values() for path in directory.rglob("*") if path.is_file() and path.name != "run_manifest.json"}
    write_json(out["artifacts"] / "run_manifest.json", {"run_id": cfg["run_id"], "status": "complete",
               "analysis_date_client": cfg["analysis_date_client"], "finished_at_machine_clock": now(),
               "config_sha256": lock["input_sha256"]["configs/mechanism.yaml"], "output_sha256": outputs})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mechanism.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=["prepare", "all", "verify"], default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        cfg, out, lock, X, splits = initialize(ROOT / args.config, args.run_id)
        print(f"v3 protocol locked; {len(lock['protected_v2_sha256'])} previous files verified", flush=True)
        if args.stage == "prepare":
            return
        if args.stage == "all" and not (out["results"] / "fits.csv").exists():
            execute(cfg, out, X, splits)
        verify(cfg, out, lock, X, splits)


if __name__ == "__main__":
    main()
