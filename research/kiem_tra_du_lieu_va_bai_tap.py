"""Reproduce the descriptive audit and toy Shapley exercise; no model training."""
from collections import Counter, defaultdict
import csv
from fractions import Fraction
from hashlib import sha256
from itertools import permutations
import json
from math import sqrt
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "nguon" / "dengue_data_Final.csv"
FEATURES = ["age", "sex", "NS1", "IgM", "IgG"]


def ranks(values):
    positions = defaultdict(list)
    for i, value in enumerate(sorted(values), 1):
        positions[value].append(i)
    return [mean(positions[value]) for value in values]


def corr(a, b):
    ma, mb = mean(a), mean(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    denominator = sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return numerator / denominator if denominator else None


def toy(x):
    a, b, c = x
    return Fraction(1, 10) + Fraction(1, 5) * a + Fraction(1, 10) * b + Fraction(1, 5) * a * b + Fraction(1, 10) * c


def enumerate_toy(background):
    x = (1, 1, 1)
    values = {mask: toy(tuple(x[j] if mask & (1 << j) else background[j] for j in range(3))) for mask in range(8)}
    contributions = []
    for order in permutations(range(3)):
        mask = 0
        row = [Fraction(0)] * 3
        for j in order:
            next_mask = mask | (1 << j)
            row[j] = values[next_mask] - values[mask]
            mask = next_mask
        contributions.append({"order": [j + 1 for j in order], "delta": [float(v) for v in row]})
    phi = [sum(Fraction(str(row["delta"][j])) for row in contributions) / 6 for j in range(3)]
    assert values[0] + sum(phi) == values[7]
    return {"background": background, "baseline": float(values[0]), "coalitions": [{"mask": mask, "S": [j + 1 for j in range(3) if mask & (1 << j)], "value": float(value)} for mask, value in values.items()], "permutations": contributions, "phi": [float(v) for v in phi], "prediction": float(values[7])}


def main():
    with PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    numeric = {c: [float(r[c]) for r in rows] for c in columns}
    profiles = defaultdict(list)
    for row in rows:
        profiles[tuple(row[c] for c in FEATURES)].append(row["Severity"])
    full = Counter(tuple(row[c] for c in columns) for row in rows)
    audit = {
        "file": str(PATH.relative_to(ROOT)), "sha256": sha256(PATH.read_bytes()).hexdigest(),
        "rows": len(rows), "columns": columns,
        "missing": {c: sum(row[c].strip().lower() in {"", "nan", "na", "null"} for row in rows) for c in columns},
        "counts": {c: dict(Counter(row[c] for row in rows)) for c in ["Severity", "sex", "NS1", "IgM", "IgG"]},
        "severe_prevalence": mean(numeric["Severity"]),
        "majority_accuracy": 1 - mean(numeric["Severity"]),
        "age": {"min": min(numeric["age"]), "max": max(numeric["age"]), "mean": mean(numeric["age"]), "median": median(numeric["age"]), "unique": len(set(numeric["age"]))},
        "duplicate_full_rows_excess": sum(n - 1 for n in full.values()),
        "duplicate_full_row_groups": sum(n > 1 for n in full.values()),
        "unique_five_feature_profiles": len(profiles),
        "repeated_five_feature_groups": sum(len(labels) > 1 for labels in profiles.values()),
        "five_feature_profiles_with_both_labels": sum(len(set(labels)) > 1 for labels in profiles.values()),
        "rows_in_conflicting_profiles": sum(len(labels) for labels in profiles.values() if len(set(labels)) > 1),
        "spearman_features": {a: {b: corr(ranks(numeric[a]), ranks(numeric[b])) for b in FEATURES} for a in FEATURES},
        "notes": ["Identical profiles do not establish repeated patients; no patient ID is released.", "Correlations are descriptive, not causal or clinical validation."],
    }
    exercises = {"original": enumerate_toy((0, 0, 0)), "alternative_background": enumerate_toy((1, 0, 0))}
    assert exercises["original"]["phi"] == [0.3, 0.2, 0.1]
    (ROOT / "audit_du_lieu.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "bai_tap_shap.json").write_text(json.dumps(exercises, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("Toy: baseline=", exercises["original"]["baseline"], "phi=", exercises["original"]["phi"], "prediction=", exercises["original"]["prediction"])
    print("Alternative background:", exercises["alternative_background"]["baseline"], exercises["alternative_background"]["phi"])


if __name__ == "__main__":
    main()
