import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .common import FEATURES, now, write_json
from .evaluate import predictive_metrics, top3
from .exact_shapley import exact_shapley
from .explain import CountingModel, explain_lime, explain_shap
from .prepare_data import background_sample


def marker(y, q, seed):
    rng = np.random.RandomState(seed)
    aligned = rng.uniform(size=len(y)) < q
    return np.where(aligned, y, 1 - y).astype(int)


def matching(y, s):
    return {"overall": float(np.mean(y == s)), "class0": float(np.mean(y[y == 0] == s[y == 0])), "class1": float(np.mean(y[y == 1] == s[y == 1]))}


def marker_benchmark(store, frame, splits, selected, backgrounds, original_rf):
    store.check_stage("E4")
    X, y = frame[FEATURES].to_numpy(dtype=float), frame.Severity.to_numpy(dtype=int)
    cfg, seeds = store.cfg["marker"], store.cfg["marker"]["rng_seeds"]
    train = splits["train"]
    models = {"C": original_rf}
    marker_rows, generation = [], {}
    for arm, q in [("R", .5), ("B", cfg["q_correlated"])]:
        st = marker(y[train], q, seeds[arm + "_train"])
        models[arm] = RandomForestClassifier(**store.cfg["model"]["rf"]).fit(np.column_stack([X[train], st]), y[train])
        joblib.dump(models[arm], store.artifacts / "models" / f"model_marker_{arm}.joblib")
        for split in ["train", "validation"]:
            ids = splits[split]
            s = marker(y[ids], q, seeds[arm + "_" + split])
            generation[arm + "_" + split] = {"seed": seeds[arm + "_" + split], "q_requested": q, "matching_actual": matching(y[ids], s)}
            marker_rows.extend({"row_id": rid, "split": split, "arm": arm, "marker": int(value)} for rid, value in zip(ids, s))
    # Dependence is assessed on validation before test marker predictions/explanations.
    dependence, validation_summary = [], []
    for arm, model in models.items():
        ids = splits["validation"]
        counter = CountingModel(model, ignore_marker=(arm == "C"))
        p0 = counter.scalar(np.column_stack([X[ids], np.zeros(len(ids))]))
        p1 = counter.scalar(np.column_stack([X[ids], np.ones(len(ids))]))
        delta = p1 - p0
        if arm == "C":
            assert np.max(abs(delta)) <= cfg["zero_tolerance"]
        validation_summary.append({"arm": arm, "n": len(ids), "mean_signed_delta": delta.mean(), "mean_abs_delta": abs(delta).mean(), "median_abs_delta": np.median(abs(delta)), "q25_abs_delta": np.quantile(abs(delta), .25), "q75_abs_delta": np.quantile(abs(delta), .75), "max_abs_delta": abs(delta).max(), "prediction_flip_fraction": np.mean((p0 >= .5) != (p1 >= .5))})
        dependence.extend({"row_id": rid, "split": "validation", "arm": arm, "p0": a, "p1": b, "delta": b - a} for rid, a, b in zip(ids, p0, p1))
    pd.DataFrame(validation_summary).to_csv(store.results / "T5_marker_validation_dependence.csv", index=False)
    write_json(store.artifacts / "marker_validation_checkpoint.json", {"validated_at": now(), "config": cfg, "summaries": validation_summary, "no_tuning_after_validation": True})
    print("E5 validation dependence", validation_summary, flush=True)
    pool = np.vstack([np.column_stack([X[train], np.zeros(len(train))]), np.column_stack([X[train], np.ones(len(train))])])
    clinical_ids = background_sample(train, y, cfg["shap_background_clinical_rows"], seeds["background"])
    B = np.vstack([np.column_stack([X[clinical_ids], np.zeros(len(clinical_ids))]), np.column_stack([X[clinical_ids], np.ones(len(clinical_ids))])])
    write_json(store.artifacts / "splits/marker_background_ids.json", {"clinical_ids": clinical_ids, "expanded_rows": len(B), "marker_values_per_clinical_row": [0, 1], "pool_rows": len(pool), "generation": generation})
    # Independent exact checks also cover the marker dummy feature in C.
    exact_rows = []
    for arm, model in models.items():
        for rid in selected["validation_pilot"]:
            x = np.r_[X[rid], 1.0]
            ex = exact_shapley(CountingModel(model, ignore_marker=(arm == "C")).scalar, x, B)
            sr = explain_shap(store, model, x, B, rid, "validation", 301, cfg["shap_budget"], "marker_balanced20", model_id=arm, context="marker")
            err = float(np.max(abs(np.asarray(sr["attributions"]) - ex["attributions"])))
            assert err <= store.cfg["explain"]["exact_tolerance"] and sr["full_enumeration"]
            if arm == "C":
                assert abs(ex["attributions"][-1]) <= cfg["zero_tolerance"]
            exact_rows.append({"arm": arm, "row_id": rid, "max_absolute_error": err, "marker_exact": ex["attributions"][-1], "marker_kernel_shap": sr["attributions"][-1], "additivity_error": sr["additivity_error"], "full_enumeration": sr["full_enumeration"]})
    pd.DataFrame(exact_rows).to_csv(store.results / "marker_exact_verification.csv", index=False)
    outcomes = []
    for arm, model in models.items():
        counter = CountingModel(model, ignore_marker=(arm == "C"))
        for rid in selected["stability"]:
            x = np.r_[X[rid], 1.0]
            p0, p1 = counter.scalar(np.vstack([np.r_[X[rid], 0.0], x]))
            dependence.append({"row_id": rid, "split": "test_stability", "arm": arm, "p0": p0, "p1": p1, "delta": p1 - p0})
            lr = explain_lime(store, model, x, pool, rid, "test", 301, 2000, model_id=arm, context="marker")
            sr = explain_shap(store, model, x, B, rid, "test", 301, cfg["shap_budget"], "marker_balanced20", model_id=arm, context="marker")
            for r in [lr, sr]:
                a = np.abs(np.asarray(r["attributions"]))
                ordering = sorted(range(6), key=lambda j: (-a[j], j))
                top = top3(r["attributions"], store.cfg["explain"]["epsilon"])
                outcomes.append({"row_id": rid, "arm": arm, "explainer": r["explainer"], "delta": p1 - p0, "abs_delta": abs(p1 - p0), "marker_attribution": r["attributions"][-1], "marker_mass": a[-1] / a.sum() if a.sum() > cfg["zero_tolerance"] else np.nan, "marker_rank": ordering.index(5) + 1, "marker_in_top3": 5 in top if top else None, "marker_near_zero": a[-1] <= store.cfg["explain"]["epsilon"], "predicted_probability_s1": r["predicted_probability"], "lime_r2": r.get("lime_r2"), "shap_additivity_error": r.get("additivity_error")})
        print("E5 explanations completed", arm, flush=True)
    pd.DataFrame(dependence).to_csv(store.results / "marker_dependence.csv", index=False)
    pd.DataFrame(outcomes).to_csv(store.results / "marker_explanations.csv", index=False)
    environment_metrics, environment_predictions = [], []
    ids = splits["test"]
    for name, q in [("q095", .95), ("q050", .5), ("q005", .05)]:
        st = marker(y[ids], q, seeds["test_" + name])
        actual = matching(y[ids], st)
        for arm, model in models.items():
            p = CountingModel(model, ignore_marker=(arm == "C")).scalar(np.column_stack([X[ids], st]))
            environment_metrics.append({"arm": arm, "environment": name, "q_requested": q, "matching_actual": actual["overall"], "matching_class0": actual["class0"], "matching_class1": actual["class1"], **predictive_metrics(y[ids], p)})
            environment_predictions.extend({"row_id": rid, "y_true": yy, "marker": ss, "arm": arm, "environment": name, "predicted_probability": pp} for rid, yy, ss, pp in zip(ids, y[ids], st, p))
    pd.DataFrame(environment_metrics).to_csv(store.results / "T5_marker_environment_metrics.csv", index=False)
    pd.DataFrame(environment_predictions).to_csv(store.results / "marker_environment_predictions.csv", index=False)
    pd.DataFrame(marker_rows).to_csv(store.results / "marker_generated_training_validation.csv", index=False)
    store.stage("E5", "complete", {"marker_explanation_design_runs": 60, "exact_validation_runs": len(exact_rows), "same_clinical_test_rows_across_environments": True})
