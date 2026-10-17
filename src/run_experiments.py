import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from .common import FEATURES, ROOT, Store, now, read_json, sha, write_json
from .evaluate import bootstrap_metrics, jaccard, near_tie, predictive_metrics, stability_group
from .exact_shapley import exact_shapley, verify_toy
from .explain import CountingModel, explain_lime, explain_shap
from .prepare_data import load_prepared, prepare
from .train import load_models, train_baselines


def export_explanations(store):
    records = [read_json(p) for p in sorted((store.results / "explanation_cache").glob("*.json"))]
    rows, metadata = [], []
    for r in records:
        key = store.cache_key(r)
        simple = {k: v for k, v in r.items() if k not in {"attributions", "conditions", "feature_names", "input", "mask_matrix", "warnings", "age_bins"}}
        simple["explanation_id"] = key
        simple["warnings"] = " | ".join(r.get("warnings", []))
        metadata.append(simple)
        for j, name in enumerate(r["feature_names"]):
            rows.append({"explanation_id": key, **{k: r[k] for k in ["run_id", "context", "row_id", "split", "model_id", "explainer", "seed", "budget_requested", "background_id", "runtime_seconds", "predicted_probability"]}, "feature_id": j, "feature_name": name, "input_value": r["input"][j], "condition": r["conditions"][j], "attribution": r["attributions"][j]})
    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(store.results / "explanations_all.csv", index=False)
    if len(all_rows):
        main_mask = all_rows.context.eq("main") & all_rows.split.eq("test") & all_rows.seed.eq(301) & (
            (all_rows.explainer.eq("LIME") & all_rows.budget_requested.eq(2000)) |
            (all_rows.explainer.eq("KernelSHAP") & all_rows.budget_requested.eq(32) & all_rows.background_id.astype(str).eq("201"))
        )
        all_rows.loc[main_mask].to_csv(store.results / "explanations_main.csv", index=False)
    else:
        all_rows.to_csv(store.results / "explanations_main.csv", index=False)
    pd.DataFrame(metadata).to_csv(store.results / "explanation_diagnostics.csv", index=False)
    return records


def pilot(store, frame, splits, selected, backgrounds, model):
    verify_toy()
    X = frame[FEATURES].to_numpy(dtype=float)
    training = X[splits["train"]]
    B = X[backgrounds["201"]]
    results, exact_rows = [], []
    for row_id in selected["validation_pilot"]:
        lr = explain_lime(store, model, X[row_id], training, row_id, "validation", 301, 2000, pilot=True)
        sr = explain_shap(store, model, X[row_id], B, row_id, "validation", 301, 32, "201")
        exact = exact_shapley(CountingModel(model).scalar, X[row_id], B)
        errors = np.abs(np.asarray(sr["attributions"]) - exact["attributions"])
        result = {"row_id": row_id, "y_true": int(frame.Severity.iloc[row_id]), "max_absolute_error": float(errors.max()), "mean_absolute_error": float(errors.mean()), "baseline_error": abs(sr["baseline"] - exact["baseline"]), "additivity_error": sr["additivity_error"], "exact_additivity_error": exact["additivity_error"], "full_enumeration": sr["full_enumeration"], "effective_features": sr["effective_features"], "mask_count": sr["unique_masks"], "lime_r2": lr["lime_r2"], "lime_point_error": lr["point_error"], "shap_seconds": sr["runtime_seconds"], "lime_seconds": lr["runtime_seconds"]}
        results.append(result)
        for j, name in enumerate(FEATURES):
            exact_rows.append({"row_id": row_id, "feature_id": j, "feature_name": name, "exact": exact["attributions"][j], "kernel_shap": sr["attributions"][j], "absolute_error": errors[j]})
        write_json(store.artifacts / f"exact_validation_{row_id}.json", exact)
        print("E2 pilot", row_id, "exact error", result["max_absolute_error"], "LIME R2", result["lime_r2"], flush=True)
    pd.DataFrame(results).to_csv(store.results / "T3_exact_verification.csv", index=False)
    pd.DataFrame(exact_rows).to_csv(store.results / "exact_validation_features.csv", index=False)
    tol = store.cfg["explain"]["exact_tolerance"]
    assert all(r["max_absolute_error"] <= tol and r["baseline_error"] <= tol and r["additivity_error"] <= tol and r["full_enumeration"] for r in results)
    store.stage("E2", "complete", {"max_exact_error": max(r["max_absolute_error"] for r in results), "validation_ids": selected["validation_pilot"]})
    write_json(store.artifacts / "configuration_locked_before_test.json", {"locked_at": now(), "config_sha256": store.config_hash, "models": read_json(store.artifacts / "models/model_metadata.json"), "case_ids_sha256": sha(store.artifacts / "splits/explanation_ids.json"), "background_ids_sha256": sha(store.artifacts / "splits/background_ids.json"), "test_predictions_created": False})


def main_explanations(store, frame, splits, selected, backgrounds, models):
    store.check_stage("E2")
    X, y = frame[FEATURES].to_numpy(dtype=float), frame.Severity.to_numpy(dtype=int)
    ids = splits["test"]
    predicted = pd.DataFrame({"row_id": ids, "y_true": y[ids]})
    metrics, predictions = [], {}
    for name, model in models.items():
        p = model.predict_proba(X[ids])[:, int(np.flatnonzero(model.classes_ == 1)[0])]
        predictions[name] = p
        predicted[name] = p
        metrics.append({"model_id": name, "split": "test", **predictive_metrics(y[ids], p)})
    predicted.to_csv(store.results / "predictions_test.csv", index=False)
    pd.DataFrame(metrics).to_csv(store.results / "T2_test_metrics.csv", index=False)
    if "test_first_opened_at" not in store.manifest:
        store.manifest["test_first_opened_at"] = now()
        store.save()
    samples, intervals = bootstrap_metrics(y[ids], predictions, store.cfg["uncertainty"]["bootstrap_replicates"], store.cfg["uncertainty"]["bootstrap_seed"])
    samples.to_csv(store.results / "bootstrap_samples.csv", index=False)
    intervals.to_csv(store.results / "bootstrap_intervals.csv", index=False)
    rf = models["RF_seed11"]
    rows = []
    eps = store.cfg["explain"]["epsilon"]
    for row_id in selected["main"]:
        lime = explain_lime(store, rf, X[row_id], X[splits["train"]], row_id, "test", 301, 2000)
        shp = explain_shap(store, rf, X[row_id], X[backgrounds["201"]], row_id, "test", 301, 32, "201")
        assert abs(lime["predicted_probability"] - shp["predicted_probability"]) < 1e-12
        p = lime["predicted_probability"]
        group = "TP" if y[row_id] and p >= .5 else "FN" if y[row_id] else "FP" if p >= .5 else "TN"
        rows.append({"row_id": row_id, "y_true": y[row_id], "predicted_probability": p, "predicted_label": int(p >= .5), "confusion_group": group, "top3_jaccard": jaccard(lime["attributions"], shp["attributions"], eps), "lime_r2": lime["lime_r2"], "lime_point_error": lime["point_error"], "lime_near_tie": near_tie(lime["attributions"], eps), "shap_near_tie": near_tie(shp["attributions"], eps), "shap_additivity_error": shp["additivity_error"], "lime_seconds": lime["runtime_seconds"], "shap_seconds": shp["runtime_seconds"], "lime_predicted_rows": lime["predicted_row_count"], "shap_predicted_rows": shp["predicted_row_count"]})
    pd.DataFrame(rows).to_csv(store.results / "T4_main_comparison.csv", index=False)
    export_explanations(store)
    store.stage("E3", "complete", {"test_metrics": metrics, "n_cases": len(rows)})


def stability(store, frame, splits, selected, backgrounds, model):
    store.check_stage("E3")
    X = frame[FEATURES].to_numpy(dtype=float)
    cfg, eps = store.cfg["explain"]["lime"], store.cfg["explain"]["epsilon"]
    per_case, per_feature, pairs = [], [], []
    for budget in cfg["budgets"]:
        for row_id in selected["stability"]:
            records = [explain_lime(store, model, X[row_id], X[splits["train"]], row_id, "test", seed, budget) for seed in cfg["seeds"]]
            group, values, js, signs, flips = stability_group(records, eps)
            per_case.append({"row_id": row_id, "budget": budget, "n_seeds": len(records), **group})
            for (a, b), score in zip(combinations(cfg["seeds"], 2), js):
                pairs.append({"row_id": row_id, "budget": budget, "seed_a": a, "seed_b": b, "jaccard": score})
            for j, name in enumerate(FEATURES):
                per_feature.append({"row_id": row_id, "budget": budget, "feature_id": j, "feature_name": name, "mean": values[:, j].mean(), "sd": values[:, j].std(ddof=1), "median": np.median(values[:, j]), "q25": np.quantile(values[:, j], .25), "q75": np.quantile(values[:, j], .75), "range": np.ptp(values[:, j]), "near_zero_runs": int((signs[:, j] == 0).sum()), "opposite_sign_pairs": int(flips[:, j].sum()), "n_pairs": len(flips)})
        print("E4 LIME completed budget", budget, flush=True)
    pd.DataFrame(per_case).to_csv(store.results / "stability_lime.csv", index=False)
    pd.DataFrame(per_feature).to_csv(store.results / "stability_lime_features.csv", index=False)
    pd.DataFrame(pairs).to_csv(store.results / "stability_lime_pairs.csv", index=False)
    bgrows, bgfeatures = [], []
    for row_id in selected["stability"]:
        records = {key: explain_shap(store, model, X[row_id], X[backgrounds[key]], row_id, "test", 301, 32, key) for key in backgrounds}
        for a, b in combinations(records, 2):
            ra, rb = records[a], records[b]
            diff = np.asarray(ra["attributions"]) - rb["attributions"]
            bgrows.append({"row_id": row_id, "background_a": a, "background_b": b, "baseline_a": ra["baseline"], "baseline_b": rb["baseline"], "baseline_shift_absolute": abs(ra["baseline"] - rb["baseline"]), "attribution_l1_shift": abs(diff).sum(), "jaccard": jaccard(ra["attributions"], rb["attributions"], eps), "all_full_enumeration": ra["full_enumeration"] and rb["full_enumeration"]})
        values = np.asarray([r["attributions"] for r in records.values()])
        for j, name in enumerate(FEATURES):
            bgfeatures.append({"row_id": row_id, "feature_id": j, "feature_name": name, "median": np.median(values[:, j]), "q25": np.quantile(values[:, j], .25), "q75": np.quantile(values[:, j], .75), "range": np.ptp(values[:, j]), "sd": np.std(values[:, j], ddof=1)})
    pd.DataFrame(bgrows).to_csv(store.results / "sensitivity_shap_background.csv", index=False)
    pd.DataFrame(bgfeatures).to_csv(store.results / "sensitivity_shap_features.csv", index=False)
    export_explanations(store)
    store.stage("E4", "complete", {"lime_design_runs": 150, "shap_background_design_runs": 30, "cases": len(selected["stability"])})


def small_shap_budgets(store, frame, selected, backgrounds, model):
    X = frame[FEATURES].to_numpy(dtype=float)
    rows = []
    for budget in store.cfg["explain"]["shap"]["small_budgets"]:
        stopped = False
        for row_id in selected["validation_pilot"]:
            exact = read_json(store.artifacts / f"exact_validation_{row_id}.json")
            for seed in store.cfg["explain"]["shap"]["small_budget_seeds"]:
                r = explain_shap(store, model, X[row_id], X[backgrounds["201"]], row_id, "validation", seed, budget, "201")
                max_error = float(np.max(np.abs(np.asarray(r["attributions"]) - exact["attributions"])))
                bad_warning = any(any(k in w.lower() for k in ["singular", "rank", "ill-conditioned"]) for w in r["warnings"])
                rows.append({"row_id": row_id, "budget": budget, "seed": seed, "max_absolute_error": max_error, "additivity_error": r["additivity_error"], "mask_count": r["unique_masks"], "full_enumeration": r["full_enumeration"], "rank_warning": bad_warning, "warnings": " | ".join(r["warnings"]), "runtime_seconds": r["runtime_seconds"]})
                if bad_warning:
                    stopped = True
                    break
            if stopped:
                break
        print("E4c budget", budget, "stopped for rank warning" if stopped else "complete", flush=True)
    pd.DataFrame(rows).to_csv(store.results / "shap_small_budget.csv", index=False)
    export_explanations(store)
    store.stage("E4c", "complete", {"runs_recorded": len(rows), "stopped_budgets": sorted({r['budget'] for r in rows if r['rank_warning']})})


def additional_rf(store, frame, splits):
    X, y = frame[FEATURES].to_numpy(dtype=float), frame.Severity.to_numpy(dtype=int)
    rows = []
    for seed in store.cfg["model"]["additional_rf_seeds"]:
        cfg = dict(store.cfg["model"]["rf"], random_state=seed)
        model = RandomForestClassifier(**cfg).fit(X[splits["train"]], y[splits["train"]])
        for split in ["validation", "test"]:
            ids = splits[split]
            p = model.predict_proba(X[ids])[:, int(np.flatnonzero(model.classes_ == 1)[0])]
            rows.append({"model_id": "RF_seed" + str(seed), "split": split, **predictive_metrics(y[ids], p)})
    pd.DataFrame(rows).to_csv(store.results / "additional_rf_seed_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="dengue_20261006_v1")
    parser.add_argument("--stage", choices=["pilot", "core", "marker", "all", "report", "verify"], default="all")
    args = parser.parse_args()
    store = Store(args.run_id)
    try:
        with threadpool_limits(limits=1):
            store.environment()
            if args.stage in {"pilot", "core", "all"}:
                if store.manifest["stages"].get("E0", {}).get("status") == "complete":
                    frame, splits, selected, backgrounds = load_prepared(store)
                else:
                    frame, splits, selected, backgrounds = prepare(store)
                models = load_models(store) if store.manifest["stages"].get("E1", {}).get("status") == "complete" else train_baselines(store, frame, splits)
                if store.manifest["stages"].get("E2", {}).get("status") != "complete":
                    pilot(store, frame, splits, selected, backgrounds, models["RF_seed11"])
                export_explanations(store)
                if args.stage != "pilot":
                    main_explanations(store, frame, splits, selected, backgrounds, models)
                    stability(store, frame, splits, selected, backgrounds, models["RF_seed11"])
                    small_shap_budgets(store, frame, selected, backgrounds, models["RF_seed11"])
                    additional_rf(store, frame, splits)
            if args.stage in {"marker", "all"}:
                store.check_stage("E4")
                from .marker_benchmark import marker_benchmark
                marker_benchmark(store, *load_prepared(store), load_models(store)["RF_seed11"])
                export_explanations(store)
            if args.stage in {"report", "all", "core", "marker"}:
                from .reporting import build_report
                build_report(store)
            if args.stage in {"verify", "all"}:
                from .verify import verify_results
                verify_results(store)
                if args.stage == "all":
                    build_report(store)
            stages = store.manifest["stages"]
            completed = all(stages.get(s, {}).get("status") == "complete" for s in ["E0", "E1", "E2", "E3", "E4", "E5", "report", "verification"])
            store.manifest["status"] = "complete" if completed else args.stage + "_complete"
            store.manifest["updated_at"] = now()
            store.manifest["implementation_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src").glob("*.py")}
            store.save()
            print("Finished", args.stage, args.run_id, flush=True)
    except Exception as exc:
        store.manifest["status"] = "failed"
        store.manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "at": now()}
        store.save()
        raise


if __name__ == "__main__":
    main()
