from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, precision_score, roc_auc_score


def predictive_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    assert p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "severe": int(y.sum()), "prevalence": float(y.mean()), "threshold": 0.5, "auroc": roc_auc_score(y, p), "ap": average_precision_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, pred), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "precision": precision_score(y, pred, zero_division=0), "brier": brier_score_loss(y, p), "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}


def top3(a, epsilon=1e-6):
    a = np.abs(np.asarray(a))
    if np.max(a) <= epsilon:
        return None
    # Exact ties resolved by stable feature order; near ties are flagged separately.
    return tuple(sorted(range(len(a)), key=lambda j: (-a[j], j))[:3])


def near_tie(a, epsilon=1e-6):
    ordered = np.sort(np.abs(a))[::-1]
    return bool(len(ordered) > 3 and abs(ordered[2] - ordered[3]) <= epsilon)


def jaccard(a, b, epsilon=1e-6):
    aa, bb = top3(a, epsilon), top3(b, epsilon)
    if aa is None or bb is None:
        return np.nan
    return len(set(aa) & set(bb)) / len(set(aa) | set(bb))


def stability_group(records, epsilon):
    values = np.asarray([r["attributions"] for r in records])
    js = [jaccard(a, b, epsilon) for a, b in combinations(values, 2)]
    signs = np.where(np.abs(values) <= epsilon, 0, np.sign(values))
    flip_pairs = np.asarray([(a * b < 0) for a, b in combinations(signs, 2)])
    return {"jaccard_mean": float(np.nanmean(js)) if not np.isnan(js).all() else np.nan, "jaccard_min": float(np.nanmin(js)) if not np.isnan(js).all() else np.nan, "jaccard_median": float(np.nanmedian(js)) if not np.isnan(js).all() else np.nan, "mean_feature_sd": float(np.mean(np.std(values, axis=0, ddof=1))), "max_feature_sd": float(np.max(np.std(values, axis=0, ddof=1))), "near_tie_runs": sum(near_tie(a, epsilon) for a in values), "sign_flip_feature_pairs": int(flip_pairs.sum()), "median_r2": float(np.nanmedian([r.get('lime_r2', np.nan) for r in records])), "median_point_error": float(np.median([r.get('point_error', 0) for r in records])), "median_runtime_seconds": float(np.median([r['runtime_seconds'] for r in records]))}, values, js, signs, flip_pairs


def bootstrap_metrics(y, predictions, repeats=1000, seed=701):
    rng = np.random.RandomState(seed)
    pools = [np.flatnonzero(y == c) for c in [0, 1]]
    rows = []
    for rep in range(repeats):
        ix = np.concatenate([rng.choice(p, size=len(p), replace=True) for p in pools])
        row = {"replicate": rep}
        for model, p in predictions.items():
            row[model + "_auroc"] = roc_auc_score(y[ix], p[ix])
            row[model + "_ap"] = average_precision_score(y[ix], p[ix])
        rows.append(row)
    samples = pd.DataFrame(rows)
    intervals = []
    for c in samples.columns[1:]:
        model, metric = c.rsplit("_", 1)
        lo, hi = samples[c].quantile([0.025, 0.975])
        intervals.append({"model_id": model, "metric": metric, "lower95": lo, "upper95": hi, "repeats": repeats, "seed": seed, "unit": "stratified test row; conditional on fixed models/split"})
    for metric in ["auroc", "ap"]:
        diff = samples["RF_seed11_" + metric] - samples["LR_" + metric]
        lo, hi = diff.quantile([0.025, 0.975])
        intervals.append({"model_id": "RF_seed11_minus_LR", "metric": metric, "lower95": lo, "upper95": hi, "repeats": repeats, "seed": seed, "unit": "paired stratified test row"})
    return samples, pd.DataFrame(intervals)
