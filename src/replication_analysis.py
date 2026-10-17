"""Exploratory Stage 4: second model, repeated splits and a perturbation plausibility audit."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
import hashlib
import importlib.metadata
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import yaml

from .common import FEATURES, ROOT, now, read_json, sha, write_json
from .evaluate import jaccard, predictive_metrics
from .exact_shapley import exact_shapley
from .mechanism_analysis import cell_ids, cell_means
from .prepare_data import background_sample, class_sample
from .supplementary_lime import draw, encode, make_explainer, score, weights

CONFIG = ROOT / "configs/replication.yaml"


def make_splits(y, cfg, seed):
    ids = np.arange(len(y))
    remaining, test = train_test_split(ids, test_size=cfg["test_fraction"], stratify=y, random_state=seed)
    train, val = train_test_split(remaining, test_size=cfg["validation_fraction_of_remainder"],
                                  stratify=y[remaining], random_state=seed)
    return {"train": sorted(train.tolist()), "validation": sorted(val.tolist()), "test": sorted(test.tolist())}


def make_backgrounds(train, y, cfg):
    backgrounds = {}
    for seed in cfg["background_seeds"]:
        small = background_sample(train, y, 20, seed)
        remaining = sorted(set(train) - set(small))
        desired_severe = int(round(100 * y[train].mean()))
        additions = class_sample(remaining, y, [100 - desired_severe - int((y[small] == 0).sum()),
                                              desired_severe - int(y[small].sum())],
                                 seed + cfg["background_extension_seed_offset"])
        backgrounds[f"n20_seed{seed}"] = small
        backgrounds[f"n100_seed{seed}"] = sorted(small + additions)
    backgrounds["full"] = list(train)
    return backgrounds


def make_model(name, params):
    return (RandomForestClassifier if name == "RF" else GradientBoostingClassifier)(**params)


def lime_config(model, X, train, cases, cfg, saved_dir=None):
    predict = lambda rows: model.predict_proba(rows)[:, 1]
    w0 = cfg["width_multiplier"] * np.sqrt(5)
    training = X[train]
    records = []
    for row in cases:
        x = X[row]
        for seed in cfg["fit_seeds"]:
            exq = make_explainer(training, seed, "quartile", w0)
            exc = make_explainer(training, seed, "continuous", w0)
            fit_raw, fit_z, fit_d = draw(exq, x, cfg["fit_budget"])
            eval_ex = make_explainer(training, seed + cfg["eval_seed_offset"], "quartile", w0)
            eval_raw, eval_z, eval_d = draw(eval_ex, x, cfg["eval_budget"], exclude_forced_x=True)
            if saved_dir is not None:
                with np.load(saved_dir / f"quartile_row{row}_seed{seed}.npz") as saved:
                    assert np.array_equal(saved["fit_raw"], fit_raw) and np.array_equal(saved["eval_raw"], eval_raw)
                    assert np.allclose(saved["fit_y"], predict(fit_raw), atol=1e-12, rtol=0)
            fit_y, eval_y = predict(fit_raw), predict(eval_raw)
            fit_w, eval_w = weights(fit_d, w0), weights(eval_d, w0)
            means, _, _ = cell_means(eval_z, eval_y, eval_w)
            floor = float(np.average((eval_y - means[cell_ids(eval_z)]) ** 2, weights=eval_w))
            arms = {"quartile": (fit_z, eval_z, encode(exq, x[None, :], x)),
                    "continuous": (encode(exc, fit_raw, x), encode(exc, eval_raw, x), encode(exc, x[None, :], x))}
            for arm, (zf, ze, zx) in arms.items():
                regression = Ridge(alpha=cfg["ridge_alpha"], fit_intercept=True, random_state=seed).fit(zf, fit_y, sample_weight=fit_w)
                held = score(eval_y, regression.predict(ze), eval_w)
                record = {"row_id": row, "fit_seed": seed, "arm": arm, "heldout_rmse": held["rmse"], "heldout_r2": held["r2"],
                          "fit_r2": score(fit_y, regression.predict(zf), fit_w)["r2"],
                          "point_error": abs(float(regression.predict(zx)[0]) - float(fit_y[0]))}
                if arm == "quartile":
                    record["within_pattern_fraction"] = floor / held["rmse"] ** 2
                records.append(record)
    return pd.DataFrame(records)


def shap_config(model, X, cases, backgrounds):
    predict = lambda rows: model.predict_proba(rows)[:, 1]
    rows = []
    for row in cases:
        results = {key: exact_shapley(predict, X[row], X[ids]) for key, ids in backgrounds.items()}
        full = results["full"]["attributions"]
        for key, result in results.items():
            if key == "full":
                continue
            phi = result["attributions"]
            both = (abs(phi) > 1e-6) & (abs(full) > 1e-6)
            rows.append({"row_id": row, "background": key, "size": int(key[1:].split("_")[0]),
                         "baseline": result["baseline"], "l1_to_full": float(abs(phi - full).sum()),
                         "jaccard_with_full": jaccard(phi, full), "sign_flips": int((np.sign(phi) != np.sign(full))[both].sum()),
                         "additivity_error": result["additivity_error"]})
    return pd.DataFrame(rows), results["full"]["baseline"]


def replication(cfg, out, frame):
    X, y = frame[FEATURES].to_numpy(float), frame.Severity.to_numpy(int)
    v1 = ROOT / "artifacts" / cfg["baseline_run"]
    v2 = ROOT / "artifacts" / cfg["supplementary_run"]
    saved_splits = read_json(v1 / "splits/split_ids.json")
    saved_cases = read_json(v2 / "splits/validation_ids.json")["row_ids"]
    v3_fits = pd.read_csv(ROOT / "results" / cfg["mechanism_run"] / "fits.csv")
    summary, lime_all, shap_all = [], [], []
    for split_seed in cfg["split_seeds"]:
        splits = make_splits(y, cfg, split_seed)
        cases = class_sample(splits["validation"], y, [cfg["validation_per_class"]] * 2, cfg["validation_seed"])
        backgrounds = make_backgrounds(splits["train"], y, cfg)
        if split_seed == 42:
            assert splits == saved_splits and cases == saved_cases
            assert backgrounds["n20_seed201"] == read_json(v2 / "splits/background_ids.json")["n20_seed201"]
        for name, params in cfg["models"].items():
            model = make_model(name, params).fit(X[splits["train"]], y[splits["train"]])
            assert model.classes_.tolist() == [0, 1]
            reproduce = split_seed == 42 and name == "RF"
            if reproduce:
                saved = joblib.load(v1 / "models/model_rf_seed11.joblib")
                assert np.array_equal(saved.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1])
            test = predictive_metrics(y[splits["test"]], model.predict_proba(X[splits["test"]])[:, 1])
            val = predictive_metrics(y[splits["validation"]], model.predict_proba(X[splits["validation"]])[:, 1])
            lime = lime_config(model, X, splits["train"], cases, cfg, v2 / "neighborhoods" if reproduce else None)
            if reproduce:
                ref = v3_fits[v3_fits.arm.isin(["quartile_ridge", "continuous_same_samples"])].copy()
                ref["arm"] = ref.arm.map({"quartile_ridge": "quartile", "continuous_same_samples": "continuous"})
                merged = lime.merge(ref, on=["row_id", "fit_seed", "arm"])
                assert len(merged) == len(lime) and np.allclose(merged.heldout_rmse, merged.quartile_rmse, atol=1e-12)
            shap, full_baseline = shap_config(model, X, cases, backgrounds)
            lime["model"], lime["split_seed"], shap["model"], shap["split_seed"] = name, split_seed, name, split_seed
            lime_all.append(lime)
            shap_all.append(shap)
            per_case = lime.groupby(["arm", "row_id"]).median(numeric_only=True)
            q, c = per_case.loc["quartile"], per_case.loc["continuous"]
            base = shap.drop_duplicates("background").groupby("size").baseline.std()
            case_mean = shap.groupby(["size", "row_id"])[["l1_to_full", "jaccard_with_full", "sign_flips"]].mean()
            agg = case_mean.groupby("size").median()
            summary.append({"model": name, "split_seed": split_seed, "test_auroc": test["auroc"], "test_ap": test["ap"],
                            "test_sensitivity": test["sensitivity"], "test_severe": test["severe"], "validation_auroc": val["auroc"],
                            "quartile_rmse": q.heldout_rmse.median(), "continuous_rmse": c.heldout_rmse.median(),
                            "cases_continuous_lower": int((c.heldout_rmse < q.heldout_rmse - 1e-12).sum()),
                            "quartile_fit_r2": q.fit_r2.median(), "within_pattern_fraction": q.within_pattern_fraction.median(),
                            "quartile_point_error": q.point_error.median(), "continuous_point_error": c.point_error.median(),
                            "cases_continuous_point_lower": int((c.point_error < q.point_error - 1e-12).sum()),
                            "baseline_sd_n20": base[20], "baseline_sd_n100": base[100], "baseline_full": full_baseline,
                            "l1_n20": agg.loc[20, "l1_to_full"], "l1_n100": agg.loc[100, "l1_to_full"],
                            "jaccard_n20": agg.loc[20, "jaccard_with_full"], "jaccard_n100": agg.loc[100, "jaccard_with_full"],
                            "sign_flips_n20": agg.loc[20, "sign_flips"], "sign_flips_n100": agg.loc[100, "sign_flips"],
                            "max_additivity_error": shap.additivity_error.max(), "reproduced_stage_1_3": reproduce})
            print(f"split {split_seed} {name}: test AUROC {test['auroc']:.3f}, RMSE q {summary[-1]['quartile_rmse']:.4f} "
                  f"-> c {summary[-1]['continuous_rmse']:.4f}", flush=True)
    table = pd.DataFrame(summary)
    table.to_csv(out["results"] / "replication_summary.csv", index=False)
    pd.concat(lime_all).to_csv(out["results"] / "replication_lime_fits.csv", index=False)
    pd.concat(shap_all).to_csv(out["results"] / "replication_shap.csv", index=False)
    return table


def nearest_distance(rows, train_rows, age_sd, window):
    """Distance to the nearest training patient: |age gap| / SD + Hamming distance on the four binary inputs."""
    patterns = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)
    train_keys = (train_rows[:, 1:].astype(int) * (2 ** np.arange(4))).sum(axis=1)
    best = np.full(len(rows), np.inf)
    same_pattern_gap = np.full(len(rows), np.inf)
    row_keys = (np.rint(rows[:, 1:]).astype(int) * (2 ** np.arange(4))).sum(axis=1)
    for key, pattern in enumerate(patterns):
        ages = np.sort(train_rows[train_keys == key, 0])
        if len(ages) == 0:
            continue
        position = np.searchsorted(ages, rows[:, 0])
        right = ages[np.clip(position, 0, len(ages) - 1)]
        left = ages[np.clip(position - 1, 0, len(ages) - 1)]
        gap = np.minimum(abs(rows[:, 0] - left), abs(rows[:, 0] - right))
        hamming = abs(rows[:, 1:] - pattern).sum(axis=1)
        best = np.minimum(best, gap / age_sd + hamming)
        same_pattern_gap = np.where(row_keys == key, gap, same_pattern_gap)
    return best, same_pattern_gap > window


def audit(cfg, out, frame):
    X = frame[FEATURES].to_numpy(float)
    splits = read_json(ROOT / "artifacts" / cfg["baseline_run"] / "splits/split_ids.json")
    v2 = ROOT / "artifacts" / cfg["supplementary_run"]
    cases = read_json(v2 / "splits/validation_ids.json")["row_ids"]
    backgrounds = read_json(v2 / "splits/background_ids.json")
    train = X[splits["train"]]
    age_min, age_max, age_sd = train[:, 0].min(), train[:, 0].max(), train[:, 0].std()
    sources = {"Real validation patients": X[splits["validation"]]}
    for representation, label in [("quartile", "LIME quartile sampler"), ("continuous", "LIME Gaussian-age sampler")]:
        sources[label] = np.vstack([np.load(v2 / "neighborhoods" / f"{representation}_row{row}_seed{seed}.npz")["fit_raw"][1:]
                                    for row in cases for seed in cfg["fit_seeds"]])
    for size, label in [("n20", "SHAP hybrids, 20-row backgrounds"), ("n100", "SHAP hybrids, 100-row backgrounds"),
                        ("full_train478", "SHAP hybrids, full training set")]:
        hybrids = []
        for key, ids in backgrounds.items():
            if not key.startswith(size):
                continue
            for row in cases:
                for mask in range(1, 31):
                    rows = X[ids].copy()
                    for j in range(5):
                        if mask & (1 << j):
                            rows[:, j] = X[row, j]
                    hybrids.append(rows)
        sources[label] = np.vstack(hybrids)
    records = []
    for label, rows in sources.items():
        distance, no_match = nearest_distance(rows, train, age_sd, cfg["audit_age_window_years"])
        records.append({"source": label, "rows": len(rows),
                        "age_outside_train_range": float(np.mean((rows[:, 0] < age_min) | (rows[:, 0] > age_max))),
                        "non_integer_age": float(np.mean(abs(rows[:, 0] - np.rint(rows[:, 0])) > 1e-9)),
                        "median_nearest_distance": float(np.median(distance)),
                        "p90_nearest_distance": float(np.quantile(distance, 0.9)),
                        "no_same_pattern_patient_within_window": float(np.mean(no_match))})
    table = pd.DataFrame(records)
    table.to_csv(out["results"] / "plausibility_audit.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "replication", "audit"], default="all")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.run_id:
        cfg["run_id"] = args.run_id
    out = {name: ROOT / name / cfg["run_id"] for name in ["artifacts", "results"]}
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    # Git with core.autocrlf may check the CSV out with CRLF; the checksum refers to the LF bytes.
    data_bytes = (ROOT / cfg["data_path"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data_bytes).hexdigest() == cfg["data_sha256"]
    lock = out["artifacts"] / "locked_before_v4.json"
    inputs = [CONFIG, ROOT / cfg["protocol_file"], ROOT / "src/replication_analysis.py"]
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in inputs}
    if not lock.exists():
        write_json(lock, {"run_id": cfg["run_id"], "locked_at_machine_clock": now(), "exploratory_after_stages_1_3": True,
                          "config": cfg, "input_sha256": hashes,
                          "packages": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "lime", "shap"]},
                          "python": sys.version})
    frame = pd.read_csv(ROOT / cfg["data_path"])
    if args.stage in ["all", "audit"]:
        print(audit(cfg, out, frame).to_string(index=False))
    if args.stage in ["all", "replication"]:
        print(replication(cfg, out, frame).to_string(index=False))


if __name__ == "__main__":
    main()
