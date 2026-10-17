from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import ROOT, read_json, sha, write_json


def class_sample(ids, y, counts, seed):
    rng = np.random.RandomState(seed)
    picked = []
    for label, count in enumerate(counts):
        pool = np.asarray(ids)[y[np.asarray(ids)] == label]
        picked.extend(rng.choice(pool, size=count, replace=False).tolist())
    return sorted(picked)


def background_sample(ids, y, size, seed):
    sample, _ = train_test_split(np.asarray(ids), train_size=size, stratify=y[np.asarray(ids)], random_state=seed)
    return sorted(sample.tolist())


def load_frame(store):
    path = ROOT / store.cfg["data"]["path"]
    if sha(path) != store.cfg["data"]["sha256"]:
        raise ValueError("Raw data checksum mismatch")
    return pd.read_csv(path)


def prepare(store):
    cfg = store.cfg
    source = ROOT / cfg["data"]["source_path"]
    raw = ROOT / cfg["data"]["path"]
    if sha(source) != cfg["data"]["sha256"]:
        raise ValueError("Source CSV differs from research snapshot")
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        raw.write_bytes(source.read_bytes())
    frame = load_frame(store)
    features = cfg["data"]["features"]
    assert features == ["age", "sex", "NS1", "IgM", "IgG"]
    assert cfg["data"]["target"] not in features
    assert len(frame) == 798 and not frame.isna().any().any()
    for c in ["Severity", "sex", "NS1", "IgM", "IgG"]:
        assert set(frame[c]) == {0, 1}
    assert frame.age.between(1, 97).all()
    ids = np.arange(len(frame))
    y = frame.Severity.to_numpy(dtype=int)
    remaining, test = train_test_split(ids, test_size=cfg["split"]["test_fraction"], stratify=y, random_state=cfg["split"]["seed"])
    train, val = train_test_split(remaining, test_size=cfg["split"]["validation_fraction_of_remainder"], stratify=y[remaining], random_state=cfg["split"]["seed"])
    splits = {"train": sorted(train.tolist()), "validation": sorted(val.tolist()), "test": sorted(test.tolist())}
    for a, b in combinations(splits.values(), 2):
        assert not set(a) & set(b)
    assert set().union(*map(set, splits.values())) == set(ids)
    s = cfg["selection"]
    selected = {
        "main": class_sample(splits["test"], y, [s["main_per_class"]] * 2, s["main_seed"]),
        "validation_pilot": class_sample(splits["validation"], y, s["pilot_class_counts"], s["pilot_seed"]),
        "seeds": {"main": s["main_seed"], "stability": s["stability_seed"], "pilot": s["pilot_seed"]},
    }
    selected["stability"] = class_sample(selected["main"], y, [s["stability_per_class"]] * 2, s["stability_seed"])
    backgrounds = {str(seed): background_sample(splits["train"], y, s["background_rows"], seed) for seed in s["background_seeds"]}
    write_json(store.artifacts / "splits/split_ids.json", splits)
    write_json(store.artifacts / "splits/explanation_ids.json", selected)
    write_json(store.artifacts / "splits/background_ids.json", backgrounds)
    summaries = []
    profiles = {k: set(map(tuple, frame.loc[v, features].to_numpy())) for k, v in splits.items()}
    for k, v in splits.items():
        summaries.append({"split": k, "n": len(v), "nonsevere": int((y[v] == 0).sum()), "severe": int(y[v].sum()), "severe_prevalence": float(y[v].mean()), "unique_profiles": len(profiles[k])})
    pd.DataFrame(summaries).to_csv(store.results / "T1_split_counts.csv", index=False)
    overlap = {a + "_" + b: len(profiles[a] & profiles[b]) for a, b in combinations(splits, 2)}
    test_seen = sum(tuple(row) in profiles["train"] for row in frame.loc[splits["test"], features].to_numpy())
    audit = {"source_url": "https://raw.githubusercontent.com/joehuang1980/DVpatient/7db6fb55f7b73a6586dad7c2f7c1ed706914b5f7/dengue%20data-Final.csv", "sha256": sha(raw), "retrieved_date": "2026-10-06", "n_rows": len(frame), "columns": frame.columns.tolist(), "dtypes": frame.dtypes.astype(str).to_dict(), "missing": frame.isna().sum().to_dict(), "classes": frame.Severity.value_counts().to_dict(), "age_min": int(frame.age.min()), "age_max": int(frame.age.max()), "row_id_origin": "0-based original CSV row position; not patient ID", "profile_overlap": overlap, "test_rows_whose_profile_is_seen_in_train": test_seen, "background_summaries": {k: {"n": len(v), "severe": int(y[v].sum()), "age_mean": float(frame.loc[v, 'age'].mean()), "binary_frequencies": frame.loc[v, features[1:]].mean().to_dict()} for k, v in backgrounds.items()}}
    write_json(store.artifacts / "dataset_audit.json", audit)
    store.stage("E0", "complete", {"counts": summaries, "profile_overlap": overlap})
    return frame, splits, selected, backgrounds


def load_prepared(store):
    return (load_frame(store), read_json(store.artifacts / "splits/split_ids.json"), read_json(store.artifacts / "splits/explanation_ids.json"), read_json(store.artifacts / "splits/background_ids.json"))
