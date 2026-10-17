import time
import warnings

import numpy as np
from lime.lime_base import LimeBase
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import Ridge
import shap

from .common import FEATURES, now


class CountingModel:
    def __init__(self, model, ignore_marker=False):
        self.model = model
        self.ignore_marker = ignore_marker
        self.positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
        self.calls = 0
        self.rows = 0
        self.inputs_valid = True

    def proba(self, X):
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and np.isfinite(X).all()
        assert np.isin(X[:, 1:], [0, 1]).all(), "Categorical perturbation became fractional"
        self.calls += 1
        self.rows += len(X)
        use = X[:, :5] if self.ignore_marker else X
        result = self.model.predict_proba(use)
        assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
        return result

    def scalar(self, X):
        return self.proba(X)[:, self.positive_index]


class CapturingBase(LimeBase):
    def explain_instance_with_data(self, neighborhood_data, neighborhood_labels, distances, label, num_features, feature_selection="auto", model_regressor=None):
        if model_regressor is None:
            model_regressor = Ridge(alpha=1.0, fit_intercept=True, random_state=self.random_state)
        result = super().explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, label, num_features, feature_selection, model_regressor)
        weights = self.kernel_fn(distances)
        output = neighborhood_labels[:, label]
        fitted = model_regressor.predict(neighborhood_data)
        weighted_mean = np.average(output, weights=weights)
        sse = np.sum(weights * (output - fitted)**2)
        sst = np.sum(weights * (output - weighted_mean)**2)
        self.captured = {"data": neighborhood_data, "outputs": output, "distances": distances, "weights": weights, "r2": float(1 - sse / sst) if sst > 1e-15 else None, "weighted_rmse": float(np.sqrt(sse / weights.sum())), "effective_n": float(weights.sum()**2 / np.sum(weights**2)), "weighted_output_variance": float(sst / weights.sum())}
        return result


def make_lime(training, seed, names, cfg):
    start = time.perf_counter()
    ex = LimeTabularExplainer(
        np.asarray(training), mode="classification", feature_names=names,
        class_names=["not severe", "severe"], categorical_features=list(range(1, len(names))),
        categorical_names={1: ["female", "male"], **{j: ["negative", "positive"] for j in range(2, min(5, len(names)))}, **({5: ["s=0", "s=1"]} if len(names) == 6 else {})},
        kernel_width=cfg["width_multiplier"] * np.sqrt(len(names)),
        feature_selection=cfg["feature_selection"], discretize_continuous=True,
        discretizer=cfg["discretizer"], random_state=seed,
    )
    ex.base = CapturingBase(ex.base.kernel_fn, verbose=False, random_state=ex.random_state)
    return ex, time.perf_counter() - start


def base_metadata(store, method, row_id, split, seed, budget, background_id, model_id, context):
    return {"run_id": store.run_id, "context": context, "model_id": model_id, "explainer": method, "row_id": int(row_id), "split": split, "seed": int(seed), "budget_requested": int(budget), "background_id": str(background_id)}


def explain_lime(store, model, x, training, row_id, split, seed, budget, model_id="RF_seed11", context="main", pilot=False, use_cache=True):
    names = FEATURES + (["marker"] if len(x) == 6 else [])
    meta = base_metadata(store, "LIME", row_id, split, seed, budget, "train_stats" if len(x) == 5 else "marker_balanced_pool", model_id, context)
    cached = store.get(meta) if use_cache else None
    if cached:
        return cached
    counter = CountingModel(model, ignore_marker=(model_id == "C"))
    ex, initialization = make_lime(training, seed, names, store.cfg["explain"]["lime"])
    cfg = store.cfg["explain"]["lime"]
    regressor = Ridge(alpha=cfg["ridge_alpha"], fit_intercept=True, random_state=seed)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exp = ex.explain_instance(np.asarray(x), counter.proba, labels=(counter.positive_index,), top_labels=None, num_features=len(names), num_samples=budget, distance_metric="euclidean", model_regressor=regressor)
    elapsed = time.perf_counter() - start
    phi = np.zeros(len(names))
    conditions = [""] * len(names)
    for (fid, value), (condition, _) in zip(exp.local_exp[counter.positive_index], exp.as_list(label=counter.positive_index)):
        phi[fid], conditions[fid] = value, condition
    cap = ex.base.captured
    # A direct f(x) call is separate from the explainer budget.
    p = float(model.predict_proba(np.asarray(x)[None, :5] if model_id == "C" else np.asarray(x)[None, :])[:, counter.positive_index][0])
    local = float(np.asarray(exp.local_pred).ravel()[0])
    meta.update({"timestamp": now(), "feature_names": names, "input": np.asarray(x), "attributions": phi, "conditions": conditions, "intercept": float(exp.intercept[counter.positive_index]), "baseline": None, "predicted_probability": p, "local_pred": local, "point_error": abs(local - p), "lime_r2": cap["r2"], "lime_raw_score": float(exp.score), "weighted_rmse": cap["weighted_rmse"], "effective_n": cap["effective_n"], "weighted_output_variance": cap["weighted_output_variance"], "additivity_error": None, "runtime_seconds": elapsed, "initialization_seconds": initialization, "call_count": counter.calls, "predicted_row_count": counter.rows, "initialization_call_count": 0, "initialization_row_count": 0, "warnings": [str(w.message) for w in caught], "kernel_width": cfg["width_multiplier"] * np.sqrt(len(names)), "age_bins": ex.discretizer.names[0], "surrogate_score_data": "same neighborhood used for fitting"})
    assert abs(meta["intercept"] + phi.sum() - local) < 1e-10
    if pilot:
        np.savez_compressed(store.artifacts / f"pilot_lime_{row_id}.npz", **{k: cap[k] for k in ["data", "outputs", "distances", "weights"]})
    if use_cache:
        store.put(meta)
    return meta


def explain_shap(store, model, x, background, row_id, split, seed, budget, background_id, model_id="RF_seed11", context="main", use_cache=True):
    meta = base_metadata(store, "KernelSHAP", row_id, split, seed, budget, background_id, model_id, context)
    cached = store.get(meta) if use_cache else None
    if cached:
        return cached
    names = FEATURES + (["marker"] if len(x) == 6 else [])
    counter = CountingModel(model, ignore_marker=(model_id == "C"))
    np.random.seed(seed)
    start = time.perf_counter()
    ex = shap.KernelExplainer(counter.scalar, np.asarray(background), link="identity")
    initialization = time.perf_counter() - start
    init_calls, init_rows = counter.calls, counter.rows
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = ex.shap_values(np.asarray(x)[None, :], nsamples=budget, l1_reg=0.0, silent=True)
    elapsed = time.perf_counter() - start
    phi = np.asarray(output).reshape(-1)
    assert phi.shape == (len(names),)
    p = float(np.asarray(ex.fx).reshape(-1)[0])
    baseline = float(np.asarray(ex.expected_value).reshape(-1)[0])
    M = int(ex.M)
    used = int(ex.nsamplesAdded) if M > 1 else 0
    masks = np.asarray(ex.maskMatrix[:used]) if used else np.empty((0, M))
    count_unique = len(np.unique(masks, axis=0)) if used else 0
    full = count_unique == max(0, 2**M - 2)
    meta.update({"timestamp": now(), "feature_names": names, "input": np.asarray(x), "attributions": phi, "conditions": names, "intercept": None, "baseline": baseline, "predicted_probability": p, "local_pred": None, "lime_r2": None, "point_error": None, "additivity_error": abs(p - baseline - phi.sum()), "runtime_seconds": elapsed, "initialization_seconds": initialization, "call_count": counter.calls - init_calls, "predicted_row_count": counter.rows - init_rows, "initialization_call_count": init_calls, "initialization_row_count": init_rows, "background_rows": len(background), "effective_features": M, "effective_feature_ids": np.asarray(ex.varyingInds).tolist(), "mask_count": used, "unique_masks": count_unique, "full_enumeration": full, "mask_matrix": masks, "seed_policy": "numpy.random.seed before KernelExplainer and shap_values", "warnings": [str(w.message) for w in caught]})
    if use_cache:
        store.put(meta)
    return meta
