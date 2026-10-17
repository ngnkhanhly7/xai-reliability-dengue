import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
from itertools import combinations
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import FEATURES, ROOT, Store, now, read_json, sha, write_json
from .evaluate import jaccard
from .exact_shapley import exact_shapley
from .explain import CountingModel, explain_shap
from .prepare_data import background_sample, class_sample
from .supplementary_lime import age_probe, array_hash, fit_one, neighborhood, run_lime


def initialize(store):
    cfg = store.cfg["supplementary"]
    parent_artifacts = ROOT / "artifacts" / cfg["parent_run"]
    parent_results = ROOT / "results" / cfg["parent_run"]
    frame = pd.read_csv(ROOT / store.cfg["data"]["path"])
    assert sha(ROOT / store.cfg["data"]["path"]) == store.cfg["data"]["sha256"]
    splits = read_json(parent_artifacts / "splits/split_ids.json")
    model_path = parent_artifacts / "models/model_rf_seed11.joblib"
    parent_lock = read_json(parent_artifacts / "configuration_locked_before_test.json")
    assert sha(model_path) == parent_lock["models"]["RF_seed11"]["sha256"]
    selection = class_sample(splits["validation"], frame.Severity.to_numpy(), [cfg["validation_per_class"]] * 2, cfg["validation_seed"])
    y = frame.Severity.to_numpy()
    backgrounds = {}
    for seed in cfg["background_seeds"]:
        small = background_sample(splits["train"], y, 20, seed)
        backgrounds[f"n20_seed{seed}"] = small
        remaining = sorted(set(splits["train"]) - set(small))
        desired_severe = int(round(100 * y[splits["train"]].mean()))
        additions = class_sample(remaining, y, [100 - desired_severe - int((y[small] == 0).sum()),
                                              desired_severe - int(y[small].sum())],
                                 seed + cfg["background_extension_seed_offset"])
        backgrounds[f"n100_seed{seed}"] = sorted(small + additions)
        assert len(backgrounds[f"n100_seed{seed}"]) == 100 and set(small) <= set(backgrounds[f"n100_seed{seed}"])
    backgrounds["full_train478"] = splits["train"]
    selection_path = store.artifacts / "splits/validation_ids.json"
    bg_path = store.artifacts / "splits/background_ids.json"
    if selection_path.exists():
        assert read_json(selection_path)["row_ids"] == selection
        assert read_json(bg_path) == backgrounds
    else:
        write_json(selection_path, {"row_ids": selection, "split": "validation", "seed": cfg["validation_seed"],
                                   "per_class": cfg["validation_per_class"], "selected_at": now(), "test_v1_already_seen": True})
        write_json(bg_path, backgrounds)
    lock_path = store.artifacts / "supplementary_locked_before_execution.json"
    if not lock_path.exists():
        with zipfile.ZipFile(ROOT / "release/dengue_v1_original.zip") as archive:
            inventory = json.loads(archive.read("EVIDENCE_MANIFEST.json"))["sha256"]
        protected = {p: value for p, value in inventory.items() if p.startswith(("artifacts/", "results/", "figures/", "data/"))
                     or p in {"configs/main.yaml", "bao-cao-thuc-nghiem-lime-shap-dengue.md", "bao-cao-research-lime-shap-dengue.md", "ke-hoach-thuc-nghiem-lime-shap-dengue.md"}}
        write_json(lock_path, {"locked_at": now(), "test_v1_already_seen": True,
                              "v1_test_first_opened_at": read_json(parent_artifacts / "run_manifest.json")["test_first_opened_at"],
                              "config_sha256": store.config_hash, "model_sha256": sha(model_path),
                              "raw_sha256": sha(ROOT / store.cfg["data"]["path"]),
                              "validation_ids_sha256": sha(selection_path), "background_ids_sha256": sha(bg_path),
                              "protocol_sha256": sha(ROOT / store.cfg["protocol_file"]),
                              "implementation_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / "src").glob("*.py")},
                              "protected_v1_sha256": protected})
        store.stage("S1_locked", "complete", {"validation_ids": selection, "after_v1_test": True})
    lock = read_json(lock_path)
    assert lock["model_sha256"] == sha(model_path)
    return frame, splits, selection, backgrounds, joblib.load(model_path), parent_results


def backgrounds_experiment(store, frame, splits, selection, backgrounds, model):
    X = frame[FEATURES].to_numpy(dtype=float)
    records = []
    # Evaluate common empirical reference first; the model and all IDs are already fixed.
    for key in ["full_train478"] + [key for key in backgrounds if key != "full_train478"]:
        for row_id in selection:
            record = explain_shap(store, model, X[row_id], X[backgrounds[key]], row_id, "validation", 301, 32,
                                  key, context="v2_background_size")
            assert record["full_enumeration"] and record["additivity_error"] <= 1e-6
            records.append(record)
        print("S4 SHAP", key, "complete", flush=True)
    full = {r["row_id"]: r for r in records if r["background_id"] == "full_train478"}
    baseline = []
    for key, ids in backgrounds.items():
        example = next(r for r in records if r["background_id"] == key)
        baseline.append({"background_id": key, "size": len(ids), "seed": int(key.split("seed")[1]) if "seed" in key else None,
                         "severe": int(frame.Severity.iloc[ids].sum()), "age_mean": frame.age.iloc[ids].mean(),
                         **frame[FEATURES[1:]].iloc[ids].mean().to_dict(), "baseline": example["baseline"],
                         "baseline_abs_error_vs_full": abs(example["baseline"] - full[selection[0]]["baseline"]),
                         "ids_sha256": array_hash(np.asarray(ids))})
    comparisons, features = [], []
    eps = store.cfg["explain"]["epsilon"]
    for r in records:
        ref = full[r["row_id"]]
        phi, target = np.asarray(r["attributions"]), np.asarray(ref["attributions"])
        opposite = (phi * target < 0) & (abs(phi) > eps) & (abs(target) > eps)
        comparisons.append({"row_id": r["row_id"], "background_id": r["background_id"], "size": r["background_rows"],
                            "baseline": r["baseline"], "baseline_abs_error_vs_full": abs(r["baseline"] - ref["baseline"]),
                            "attribution_l1_vs_full": abs(phi - target).sum(), "attribution_linf_vs_full": abs(phi - target).max(),
                            "top3_jaccard_vs_full": jaccard(phi, target, eps), "sign_flips_vs_full": int(opposite.sum()),
                            "full_enumeration": r["full_enumeration"], "unique_masks": r["unique_masks"],
                            "explain_seconds": r["runtime_seconds"], "initialization_seconds": r["initialization_seconds"],
                            "predict_rows": r["predicted_row_count"], "initialization_rows": r["initialization_row_count"]})
        for j, name in enumerate(FEATURES):
            features.append({"row_id": r["row_id"], "background_id": r["background_id"], "size": r["background_rows"],
                             "feature_id": j, "feature_name": name, "phi": phi[j], "phi_full_train": target[j],
                             "absolute_error": abs(phi[j] - target[j]), "sign_flip": bool(opposite[j])})
    pd.DataFrame(baseline).to_csv(store.results / "shap_background_descriptions.csv", index=False)
    pd.DataFrame(comparisons).to_csv(store.results / "shap_vs_full_train.csv", index=False)
    pd.DataFrame(features).to_csv(store.results / "shap_feature_errors.csv", index=False)
    exact_rows = []
    for key in ["n20_seed201", "n100_seed201", "full_train478"]:
        for row_id in selection[:store.cfg["supplementary"]["exact_cases"]]:
            r = next(r for r in records if r["row_id"] == row_id and r["background_id"] == key)
            exact = exact_shapley(CountingModel(model).scalar, X[row_id], X[backgrounds[key]])
            error = np.max(abs(np.asarray(r["attributions"]) - exact["attributions"]))
            assert error <= store.cfg["explain"]["exact_tolerance"]
            exact_rows.append({"row_id": row_id, "background_id": key, "size": len(backgrounds[key]),
                               "max_absolute_error": error, "baseline_error": abs(r["baseline"] - exact["baseline"])})
            write_json(store.artifacts / f"exact_{key}_row{row_id}.json", exact)
    pd.DataFrame(exact_rows).to_csv(store.results / "shap_exact_verification.csv", index=False)
    store.stage("S4_SHAP", "complete", {"explanations": len(records), "exact_checks": len(exact_rows),
                 "validation_cases": len(selection), "full_train_is_empirical_reference": True})


def sign_flips_v1(store, parent_results):
    stable = pd.read_csv(parent_results / "stability_lime_features.csv")
    records = [read_json(p) for p in (parent_results / "explanation_cache").glob("*.json")]
    row_ids = sorted(stable.row_id.unique())
    features, flips = [], []
    eps = store.cfg["explain"]["epsilon"]
    for row_id in row_ids:
        for budget in [100, 500, 2000]:
            group = sorted((r for r in records if r["context"] == "main" and r["split"] == "test" and r["explainer"] == "LIME"
                            and r["row_id"] == row_id and r["budget_requested"] == budget), key=lambda r: r["seed"])
            assert len(group) == 5
            values = np.asarray([r["attributions"] for r in group])
            median = np.median(values, axis=0)
            ordering = sorted(range(5), key=lambda j: (-abs(median[j]), j))
            for j, name in enumerate(FEATURES):
                category = "top2_by_abs_median" if j in ordering[:2] else "remaining3"
                count = 0
                for ia, ib in combinations(range(5), 2):
                    a, b = values[ia, j], values[ib, j]
                    if a * b < 0 and abs(a) > eps and abs(b) > eps:
                        count += 1
                        flips.append({"row_id": row_id, "budget": budget, "feature_name": name, "feature_group": category,
                                      "seed_a": group[ia]["seed"], "seed_b": group[ib]["seed"], "coefficient_a": a, "coefficient_b": b,
                                      "larger_absolute_coefficient": max(abs(a), abs(b)), "smaller_absolute_coefficient": min(abs(a), abs(b))})
                features.append({"row_id": row_id, "budget": budget, "feature_name": name, "feature_group": category,
                                 "median_coefficient": median[j], "sd": values[:, j].std(ddof=1),
                                 "max_abs_coefficient": abs(values[:, j]).max(), "opposite_sign_pairs": count, "n_pairs": 10})
    pd.DataFrame(features).to_csv(store.results / "v1_sign_flip_features.csv", index=False)
    pd.DataFrame(flips).to_csv(store.results / "v1_sign_flip_pairs.csv", index=False)
    store.stage("S5_sign_flips", "complete", {"uses_saved_v1_test_cache": True, "new_test_fits": 0})


def verify(store, frame, splits, selection, backgrounds, model):
    lock = read_json(store.artifacts / "supplementary_locked_before_execution.json")
    unchanged = all(sha(ROOT / path) == checksum for path, checksum in lock["protected_v1_sha256"].items())
    assert unchanged, "A protected v1 artifact changed"
    assert lock["config_sha256"] == store.config_hash
    assert lock["validation_ids_sha256"] == sha(store.artifacts / "splits/validation_ids.json")
    assert lock["background_ids_sha256"] == sha(store.artifacts / "splits/background_ids.json")
    assert lock["protocol_sha256"] == sha(ROOT / store.cfg["protocol_file"])
    assert set(selection) <= set(splits["validation"]) and not set(selection) & set(splits["test"])
    assert frame.Severity.iloc[selection].sum() == 5
    fit = pd.read_csv(store.results / "lime_ablation.csv")
    assert len(fit) == 120 and fit.split.eq("validation").all()
    quartet = fit[fit.representation.eq("quartile")]
    assert quartet.groupby(["row_id", "fit_seed"]).fit_raw_sha256.nunique().eq(1).all()
    assert quartet.groupby(["row_id", "fit_seed"]).eval_raw_sha256.nunique().eq(1).all()
    assert fit.fit_raw_sha256.ne(fit.eval_raw_sha256).all()
    assert fit.groupby(["row_id", "fit_seed"]).common_quartile_raw_sha256.nunique().eq(1).all()
    assert fit.groupby(["row_id", "fit_seed"]).common_continuous_raw_sha256.nunique().eq(1).all()
    api = fit.official_api_max_coefficient_error.dropna()
    assert len(api) == 60 and api.max() <= 1e-10
    shp = pd.read_csv(store.results / "shap_vs_full_train.csv")
    assert len(shp) == 110 and shp.full_enumeration.all()
    ex = pd.read_csv(store.results / "shap_exact_verification.csv")
    assert len(ex) == 6 and ex.max_absolute_error.max() <= 1e-6
    for seed in store.cfg["supplementary"]["background_seeds"]:
        assert set(backgrounds[f"n20_seed{seed}"]) <= set(backgrounds[f"n100_seed{seed}"]) <= set(splits["train"])
    # Fresh sampler/refit verifies both encodings, independently of the persisted NPZ/cache.
    X = frame[FEATURES].to_numpy(dtype=float)
    row_id, seed = selection[0], 301
    regenerated = {rep: neighborhood(store, model, X[splits["train"]], X[row_id], row_id, seed, rep, fresh=True)
                   for rep in ["quartile", "continuous"]}
    pools = {rep: (a, metadata) for rep, (_, a, metadata) in regenerated.items()}
    differences = {}
    for rep, (ex, a, metadata) in regenerated.items():
        fresh = fit_one(store, model, X[splits["train"]], X[row_id], row_id, seed, rep, 1., pools, ex, a, metadata, use_cache=False)
        saved = read_json(store.results / "surrogate_cache" / f"{rep}_w1_row{row_id}_seed{seed}.json")
        assert saved["fit_raw_sha256"] == fresh["fit_raw_sha256"] and saved["eval_raw_sha256"] == fresh["eval_raw_sha256"]
        difference = np.max(abs(np.asarray(saved["attributions"]) - fresh["attributions"]))
        assert difference <= 1e-12
        assert abs(fresh["heldout_common_quartile_rmse"] - saved["heldout_common_quartile_rmse"]) <= 1e-12
        differences[rep] = float(difference)
    result = {"passed": True, "verified_at": now(), "protected_v1_files": len(lock["protected_v1_sha256"]),
              "v1_unchanged": unchanged, "validation_only_ablation": True, "shared_width_perturbations": True,
              "independent_evaluation_seeds": True, "same_common_raw_evaluation_domains": True,
              "lime_fits": len(fit), "official_api_checks": len(api), "official_api_max_error": float(api.max()),
              "fresh_rerun_max_coefficient_error": differences, "shap_explanations": len(shp),
              "shap_exact_max_error": float(pd.read_csv(store.results / "shap_exact_verification.csv").max_absolute_error.max()),
              "explicit_posthoc_test_probe": 97}
    write_json(store.results / "verification.json", result)
    store.stage("verification", "complete", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="dengue_20261006_supp_v2")
    parser.add_argument("--stage", choices=["prepare", "all", "report", "verify"], default="all")
    args = parser.parse_args()
    store = Store(args.run_id, ROOT / "configs/supplementary.yaml")
    try:
        with threadpool_limits(limits=1):
            store.environment()
            frame, splits, selection, backgrounds, model, parent_results = initialize(store)
            if args.stage == "all":
                run_lime(store, frame, splits["train"], selection, model)
                age_probe(store, frame, splits["train"], selection, model)
                backgrounds_experiment(store, frame, splits, selection, backgrounds, model)
                sign_flips_v1(store, parent_results)
            if args.stage in {"all", "verify"}:
                verify(store, frame, splits, selection, backgrounds, model)
            if args.stage in {"all", "report"}:
                from .supplementary_report import build_report
                build_report(store)
            completed = all(store.manifest["stages"].get(s, {}).get("status") == "complete"
                            for s in ["S1_locked", "S2_S3_LIME", "S3_age_probe", "S4_SHAP", "S5_sign_flips", "verification", "report"])
            store.manifest["status"] = "complete" if completed else args.stage + "_complete"
            store.manifest["implementation_sha256"] = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / "src").glob("*.py")}
            store.manifest["updated_at"] = now()
            store.save()
            print("Finished supplementary", args.stage, args.run_id, flush=True)
    except Exception as exc:
        store.manifest["status"] = "failed"
        store.manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "at": now()}
        store.save()
        raise


if __name__ == "__main__":
    main()
