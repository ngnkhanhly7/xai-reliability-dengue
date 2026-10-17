"""Paired width ablation and independent evaluation for pinned LIME 0.2.0.1."""
import hashlib
import time
import warnings

import joblib
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, pairwise_distances, r2_score

from .common import FEATURES, now, read_json, write_json
from .evaluate import jaccard, near_tie
from .explain import CountingModel


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def make_explainer(training, seed, representation, width):
    return LimeTabularExplainer(
        training, mode="classification", feature_names=FEATURES, class_names=["not severe", "severe"],
        categorical_features=[1, 2, 3, 4], feature_selection="none",
        discretize_continuous=representation == "quartile", discretizer="quartile",
        sample_around_instance=False, kernel_width=width, random_state=seed,
    )


def encode(ex, raw, x):
    z = np.asarray(raw, dtype=float).copy()
    if ex.discretizer is not None:
        bins = ex.discretizer.discretize(z)
        target = ex.discretizer.discretize(x)
        z = (bins == target).astype(float)
    else:
        z[:, 1:] = (z[:, 1:] == x[1:]).astype(float)
    return (z - ex.scaler.mean_) / ex.scaler.scale_


def draw(ex, x, n, exclude_forced_x=False):
    data, raw = ex._LimeTabularExplainer__data_inverse(x, n + int(exclude_forced_x))
    z = (data - ex.scaler.mean_) / ex.scaler.scale_
    independently_encoded = encode(ex, raw, x)
    assert np.allclose(z, independently_encoded, atol=1e-12, rtol=0)
    zx = encode(ex, x[None, :], x)[0]
    distances = pairwise_distances(z, zx[None, :], metric="euclidean").ravel()
    if exclude_forced_x:
        raw, z, distances = raw[1:], z[1:], distances[1:]
    return raw, z, distances


def weights(distances, width):
    return np.exp(-np.square(distances) / (2 * width**2))


def score(y, prediction, w):
    weighted_mean = np.average(y, weights=w)
    variance = np.average((y - weighted_mean)**2, weights=w)
    rmse = float(np.sqrt(np.average((y - prediction)**2, weights=w)))
    result = {"rmse": rmse, "r2": 1 - rmse**2 / variance if variance > 1e-15 else None,
              "effective_n": float(w.sum()**2 / np.square(w).sum()),
              "weighted_output_variance": float(variance), "weight_sum": float(w.sum())}
    assert abs(rmse**2 - mean_squared_error(y, prediction, sample_weight=w)) < 1e-12
    if variance > 1e-15:
        assert abs(result["r2"] - r2_score(y, prediction, sample_weight=w)) < 1e-10
    return result


def neighborhood(store, model, training, x, row_id, seed, representation, fresh=False):
    cfg = store.cfg["supplementary"]
    directory = store.artifacts / "neighborhoods"
    directory.mkdir(exist_ok=True)
    stem = f"{representation}_row{row_id}_seed{seed}"
    path, meta_path = directory / (stem + ".npz"), directory / (stem + ".json")
    w0 = store.cfg["explain"]["lime"]["width_multiplier"] * np.sqrt(5)
    ex = make_explainer(training, seed, representation, w0)
    if path.exists() and not fresh:
        with np.load(path) as saved:
            arrays = {k: saved[k] for k in saved.files}
        return ex, arrays, read_json(meta_path)
    start = time.perf_counter()
    fit_raw, fit_z, fit_d = draw(ex, x, cfg["fit_budget"])
    eval_seed = seed + cfg["eval_seed_offset"]
    evaluation_ex = make_explainer(training, eval_seed, representation, w0)
    eval_raw, eval_z, eval_d = draw(evaluation_ex, x, cfg["eval_budget"], exclude_forced_x=True)
    generation_seconds = time.perf_counter() - start
    counter = CountingModel(model)
    start = time.perf_counter()
    fit_y, eval_y = counter.scalar(fit_raw), counter.scalar(eval_raw)
    prediction_seconds = time.perf_counter() - start
    arrays = {"fit_raw": fit_raw, "fit_z": fit_z, "fit_distances": fit_d, "fit_y": fit_y,
              "eval_raw": eval_raw, "eval_z": eval_z, "eval_distances": eval_d, "eval_y": eval_y}
    metadata = {"timestamp": now(), "row_id": row_id, "representation": representation, "fit_seed": seed,
                "evaluation_seed": eval_seed, "fit_rows": len(fit_y), "evaluation_rows": len(eval_y),
                "fit_raw_sha256": array_hash(fit_raw), "eval_raw_sha256": array_hash(eval_raw),
                "forced_original_removed_from_evaluation": True, "generation_seconds": generation_seconds,
                "prediction_seconds_shared_across_widths": prediction_seconds,
                "predict_calls": counter.calls, "predict_rows": counter.rows,
                "fit_age_outside_train_fraction": float(np.mean((fit_raw[:, 0] < training[:, 0].min()) | (fit_raw[:, 0] > training[:, 0].max()))),
                "eval_age_outside_train_fraction": float(np.mean((eval_raw[:, 0] < training[:, 0].min()) | (eval_raw[:, 0] > training[:, 0].max())))}
    assert metadata["fit_raw_sha256"] != metadata["eval_raw_sha256"]
    if not fresh:
        np.savez_compressed(path, **arrays)
        write_json(meta_path, metadata)
    return make_explainer(training, seed, representation, w0), arrays, metadata


def fit_one(store, model, training, x, row_id, seed, representation, factor, pools, ex, a, metadata, use_cache=True):
    directory = store.results / "surrogate_cache"
    directory.mkdir(exist_ok=True)
    surrogate_dir = store.artifacts / "surrogates"
    surrogate_dir.mkdir(exist_ok=True)
    stem = f"{representation}_w{factor:g}_row{row_id}_seed{seed}"
    path = directory / (stem + ".json")
    if use_cache and path.exists():
        return read_json(path)
    w0 = store.cfg["explain"]["lime"]["width_multiplier"] * np.sqrt(5)
    width = factor * w0
    w = weights(a["fit_distances"], width)
    regressor = Ridge(alpha=store.cfg["explain"]["lime"]["ridge_alpha"], fit_intercept=True, random_state=seed)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        regressor.fit(a["fit_z"], a["fit_y"], sample_weight=w)
    fit_seconds = time.perf_counter() - start
    original_z = encode(ex, x[None, :], x)
    gx = float(regressor.predict(original_z)[0])
    fx = float(a["fit_y"][0])
    own_train = score(a["fit_y"], regressor.predict(a["fit_z"]), w)
    own_eval = score(a["eval_y"], regressor.predict(a["eval_z"]), weights(a["eval_distances"], width))
    record = {"run_id": store.run_id, "row_id": row_id, "split": "validation", "model_id": "RF_seed11",
              "timestamp": now(), "representation": representation, "width_factor": factor,
              "kernel_width": width, "fit_seed": seed, "evaluation_seed": metadata["evaluation_seed"],
              "fit_raw_sha256": metadata["fit_raw_sha256"], "eval_raw_sha256": metadata["eval_raw_sha256"],
              "attributions": regressor.coef_, "intercept": regressor.intercept_,
              "predicted_probability": fx, "local_pred": gx, "point_error": abs(fx - gx),
              **{"fit_" + k: v for k, v in own_train.items()},
              **{"heldout_native_" + k: v for k, v in own_eval.items()},
              "surrogate_fit_seconds": fit_seconds,
              "fit_age_outside_train_fraction": metadata["fit_age_outside_train_fraction"],
              "eval_age_outside_train_fraction": metadata["eval_age_outside_train_fraction"],
              "near_tie": near_tie(regressor.coef_, store.cfg["explain"]["epsilon"]),
              "warnings": [str(w.message) for w in caught]}
    for domain, (pool, _) in pools.items():
        transformed = encode(ex, pool["eval_raw"], x)
        measured = score(pool["eval_y"], regressor.predict(transformed), weights(pool["eval_distances"], w0))
        record.update({"heldout_common_" + domain + "_" + k: v for k, v in measured.items()})
        record["common_" + domain + "_raw_sha256"] = array_hash(pool["eval_raw"])
    # Verify direct fits against the public explain_instance path for both encodings.
    if factor == 1:
        official = make_explainer(training, seed, representation, w0)
        official_reg = Ridge(alpha=1.0, fit_intercept=True, random_state=seed)
        exp = official.explain_instance(x, model.predict_proba, labels=(1,), top_labels=None,
                                       num_features=5, num_samples=store.cfg["supplementary"]["fit_budget"],
                                       model_regressor=official_reg)
        official_coef = np.zeros(5)
        for fid, coefficient in exp.local_exp[1]:
            official_coef[fid] = coefficient
        record["official_api_max_coefficient_error"] = float(np.max(abs(regressor.coef_ - official_coef)))
        record["official_api_point_error"] = abs(gx - float(np.asarray(exp.local_pred).ravel()[0]))
        assert record["official_api_max_coefficient_error"] <= 1e-10 and record["official_api_point_error"] <= 1e-10
    if use_cache:
        joblib.dump(regressor, surrogate_dir / (stem + ".joblib"))
        write_json(path, record)
    return record


def run_lime(store, frame, training_ids, selected, model):
    X = frame[FEATURES].to_numpy(dtype=float)
    training = X[training_ids]
    cfg = store.cfg["supplementary"]
    records = []
    for row_id in selected:
        for seed in cfg["fit_seeds"]:
            neighborhoods = {rep: neighborhood(store, model, training, X[row_id], row_id, seed, rep)
                             for rep in ["quartile", "continuous"]}
            pools = {rep: (a, meta) for rep, (_, a, meta) in neighborhoods.items()}
            for representation, (ex, a, metadata) in neighborhoods.items():
                factors = cfg["width_factors"] if representation == "quartile" else [cfg["continuous_width_factor"]]
                records.extend(fit_one(store, model, training, X[row_id], row_id, seed, representation, factor,
                                       pools, ex, a, metadata) for factor in factors)
        print("S2/S3 LIME validation row", row_id, "complete", flush=True)
    metadata = pd.DataFrame([{k: v for k, v in r.items() if k not in {"attributions", "warnings"}} |
                             {"warnings": " | ".join(r["warnings"])} for r in records])
    metadata.to_csv(store.results / "lime_ablation.csv", index=False)
    coefficients = [{"row_id": r["row_id"], "representation": r["representation"], "width_factor": r["width_factor"],
                     "fit_seed": r["fit_seed"], "feature_id": j, "feature_name": name, "coefficient": r["attributions"][j]}
                    for r in records for j, name in enumerate(FEATURES)]
    pd.DataFrame(coefficients).to_csv(store.results / "lime_coefficients.csv", index=False)
    stable = []
    from itertools import combinations
    for representation, factor in [("quartile", f) for f in cfg["width_factors"]] + [("continuous", 1.0)]:
        for row_id in selected:
            group = [r for r in records if r["representation"] == representation and r["width_factor"] == factor and r["row_id"] == row_id]
            values = np.asarray([r["attributions"] for r in group])
            scores = [jaccard(a, b) for a, b in combinations(values, 2)]
            stable.append({"row_id": row_id, "representation": representation, "width_factor": factor,
                           "n_seeds": len(group), "n_seed_pairs": len(scores), "mean_pair_jaccard": np.nanmean(scores),
                           "mean_feature_sd": np.std(values, axis=0, ddof=1).mean(),
                           "near_tie_runs": sum(r["near_tie"] for r in group)})
    pd.DataFrame(stable).to_csv(store.results / "lime_stability.csv", index=False)
    store.stage("S2_S3_LIME", "complete", {"fits": len(records), "validation_cases": len(selected),
                 "independent_eval_per_fit": cfg["eval_budget"], "official_api_checks": int(metadata.official_api_max_coefficient_error.notna().sum())})
    return records


def age_probe(store, frame, training_ids, selected, model):
    X = frame[FEATURES].to_numpy(dtype=float)
    cfg = store.cfg["supplementary"]
    ex = make_explainer(X[training_ids], 301, "quartile", .75 * np.sqrt(5))
    parent_cache = store.cfg["supplementary"]["parent_run"]
    from .common import ROOT
    old_records = [read_json(p) for p in (ROOT / "results" / parent_cache / "explanation_cache").glob("*.json")]
    rows, summary = [], []
    for row_id in selected + [cfg["age_probe_test_row"]]:
        x = X[row_id]
        all_ages = np.arange(int(X[training_ids, 0].min()), int(X[training_ids, 0].max()) + 1)
        raw = np.tile(x, (len(all_ages), 1))
        raw[:, 0] = all_ages
        target_bin = ex.discretizer.discretize(x)[0]
        same = ex.discretizer.discretize(raw)[:, 0] == target_bin
        raw = raw[same]
        z = encode(ex, raw, x)
        assert np.max(np.ptp(z, axis=0)) == 0
        if row_id in selected:
            record = read_json(store.results / "surrogate_cache" / f"quartile_w1_row{row_id}_seed301.json")
            split = "validation"
        else:
            record = next(r for r in old_records if r["row_id"] == row_id and r["context"] == "main" and r["split"] == "test"
                          and r["explainer"] == "LIME" and r["seed"] == 301 and r["budget_requested"] == 2000)
            split = "test_v1_posthoc_probe"
        prediction = CountingModel(model).scalar(raw)
        gx = z @ np.asarray(record["attributions"]) + record["intercept"]
        assert np.ptp(gx) < 1e-12
        rows.extend({"row_id": row_id, "split": split, "age": age, "target_age": x[0],
                     "rf_probability": p, "quartile_surrogate_probability": g, "age_bin": target_bin}
                    for age, p, g in zip(raw[:, 0], prediction, gx))
        summary.append({"row_id": row_id, "split": split, "age_min": raw[:, 0].min(), "age_max": raw[:, 0].max(),
                        "rf_min": prediction.min(), "rf_max": prediction.max(), "rf_range": np.ptp(prediction),
                        "minimax_grid_error_lower_bound": np.ptp(prediction) / 2, "surrogate_constant": gx[0],
                        "original_age": x[0], "original_probability": record["predicted_probability"]})
    pd.DataFrame(rows).to_csv(store.results / "age_within_bin_probe.csv", index=False)
    pd.DataFrame(summary).to_csv(store.results / "age_within_bin_summary.csv", index=False)
    store.stage("S3_age_probe", "complete", {"validation_cases": len(selected), "explicit_posthoc_test_row": cfg["age_probe_test_row"]})
