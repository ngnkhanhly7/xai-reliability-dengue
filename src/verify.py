from itertools import combinations
import re

import numpy as np
import pandas as pd

from .common import FEATURES, ROOT, read_json, sha, write_json
from .exact_shapley import verify_toy
from .explain import explain_lime, explain_shap
from .prepare_data import load_prepared
from .train import load_models


def verify_results(store):
    store.check_stage("E4")
    frame, splits, selected, backgrounds = load_prepared(store)
    models = load_models(store)
    X, y = frame[FEATURES].to_numpy(dtype=float), frame.Severity.to_numpy(dtype=int)
    verify_toy()
    for a, b in combinations(splits.values(), 2):
        assert not set(a) & set(b)
    assert sum(map(len, splits.values())) == 798
    assert len(selected["main"]) == 20 and y[selected["main"]].sum() == 10
    assert len(selected["stability"]) == 10 and y[selected["stability"]].sum() == 5
    assert set(selected["stability"]) <= set(selected["main"])
    assert all(set(v) <= set(splits["train"]) for v in backgrounds.values())
    locked = read_json(store.artifacts / "configuration_locked_before_test.json")
    assert locked["locked_at"] < store.manifest["test_first_opened_at"]
    assert locked["config_sha256"] == store.config_hash
    assert locked["case_ids_sha256"] == sha(store.artifacts / "splits/explanation_ids.json")
    assert locked["background_ids_sha256"] == sha(store.artifacts / "splits/background_ids.json")
    for name, model_meta in locked["models"].items():
        path = store.artifacts / "models" / ("model_lr.joblib" if name == "LR" else "model_rf_seed11.joblib")
        assert sha(path) == model_meta["sha256"]
    exact = pd.read_csv(store.results / "T3_exact_verification.csv")
    assert len(exact) == 5 and exact.max_absolute_error.max() <= 1e-6
    comparison = pd.read_csv(store.results / "T4_main_comparison.csv")
    assert set(comparison.row_id) == set(selected["main"])
    main_attributions = pd.read_csv(store.results / "explanations_main.csv")
    assert len(main_attributions) == 200
    assert set(main_attributions.row_id) == set(selected["main"])
    assert main_attributions.groupby(["row_id", "explainer"]).feature_id.apply(
        lambda ids: len(ids) == 5 and set(ids) == set(range(5))
    ).all()
    records = [read_json(p) for p in (store.results / "explanation_cache").glob("*.json")]
    main_shap = [r for r in records if r["context"] == "main" and r["split"] == "test" and r["explainer"] == "KernelSHAP" and r["budget_requested"] == 32]
    assert all(r["full_enumeration"] and r["additivity_error"] <= 1e-6 for r in main_shap)
    stability = pd.read_csv(store.results / "stability_lime.csv")
    assert len(stability) == 30 and stability.n_seeds.eq(5).all()
    sensitivity = pd.read_csv(store.results / "sensitivity_shap_background.csv")
    assert len(sensitivity) == 30 and sensitivity.all_full_enumeration.all()
    rid = selected["main"][0]
    fresh_lime = explain_lime(store, models["RF_seed11"], X[rid], X[splits["train"]], rid, "test", 301, 2000, use_cache=False)
    fresh_shap = explain_shap(store, models["RF_seed11"], X[rid], X[backgrounds["201"]], rid, "test", 301, 32, "201", use_cache=False)
    saved_lime, saved_shap = store.get(fresh_lime), store.get(fresh_shap)
    lime_diff = float(np.max(abs(np.asarray(fresh_lime["attributions"]) - saved_lime["attributions"])))
    shap_diff = float(np.max(abs(np.asarray(fresh_shap["attributions"]) - saved_shap["attributions"])))
    assert lime_diff <= 1e-12 and shap_diff <= 1e-12
    prediction = float(models["RF_seed11"].predict_proba(X[[rid]])[0, 1])
    assert abs(prediction - fresh_lime["predicted_probability"]) <= 1e-12
    assert abs(prediction - fresh_shap["predicted_probability"]) <= 1e-12
    missing_links = []
    report = ROOT / "bao-cao-thuc-nghiem-lime-shap-dengue.md"
    if report.exists():
        for target in re.findall(r"\]\(([^)]+)\)", report.read_text(encoding="utf-8")):
            if not target.startswith(("http", "#")) and not (ROOT / target).exists():
                missing_links.append(target)
    assert not missing_links
    if store.manifest["stages"].get("E5", {}).get("status") == "complete":
        marker_exact = pd.read_csv(store.results / "marker_exact_verification.csv")
        assert len(marker_exact) == 15 and marker_exact.max_absolute_error.max() <= 1e-6
        assert marker_exact.loc[marker_exact.arm.eq("C"), "marker_exact"].abs().max() <= 1e-12
        marker = pd.read_csv(store.results / "marker_explanations.csv")
        assert len(marker) == 60
        assert marker.groupby(["arm", "explainer"]).row_id.apply(
            lambda ids: len(ids) == 10 and set(ids) == set(selected["stability"])
        ).all()
        env = pd.read_csv(store.results / "T5_marker_environment_metrics.csv")
        c = env.loc[env.arm.eq("C"), ["auroc", "ap", "balanced_accuracy"]].to_numpy()
        assert np.max(abs(c - c[0])) <= 1e-12
    result = {"passed": True, "rerun_row_id": rid, "lime_rerun_max_absolute_difference": lime_diff, "shap_rerun_max_absolute_difference": shap_diff, "exact_max_absolute_error": float(exact.max_absolute_error.max()), "split_disjoint": True, "ids_selected_before_predictions": True, "selected_ids_unchanged": True, "frozen_models_unchanged": True, "main_feature_rows": len(main_attributions), "cached_explanations": len(records), "all_main_shap_full_enumeration": True, "source_documents_unchanged": all(sha(ROOT / p) == value for p, value in read_json(store.artifacts / "environment.json")["input_documents"].items()), "report_local_links_exist": not missing_links}
    assert result["source_documents_unchanged"]
    write_json(store.results / "verification.json", result)
    store.stage("verification", "complete", result)
    return result
