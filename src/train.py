import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import sha, write_json
from .evaluate import predictive_metrics


def train_baselines(store, frame, splits):
    train, val = splits["train"], splits["validation"]
    features = store.cfg["data"]["features"]
    X, y = frame[features].to_numpy(dtype=float), frame.Severity.to_numpy(dtype=int)
    lr = Pipeline([("preprocess", ColumnTransformer([("age", StandardScaler(), [0])], remainder="passthrough")), ("classifier", LogisticRegression(**store.cfg["model"]["lr"]))])
    rf = RandomForestClassifier(**store.cfg["model"]["rf"])
    models = {"LR": lr, "RF_seed11": rf}
    metrics, predictions = [], pd.DataFrame({"row_id": val, "y_true": y[val]})
    metadata = {}
    for name, model in models.items():
        model.fit(X[train], y[train])
        assert model.classes_.tolist() == [0, 1]
        class_index = int(np.flatnonzero(model.classes_ == 1)[0])
        p = model.predict_proba(X[val])[:, class_index]
        metrics.append({"model_id": name, "split": "validation", **predictive_metrics(y[val], p)})
        predictions[name] = p
        path = store.artifacts / "models" / ("model_lr.joblib" if name == "LR" else "model_rf_seed11.joblib")
        joblib.dump(model, path)
        metadata[name] = {"sha256": sha(path), "classes": model.classes_.tolist(), "positive_class_index": class_index, "trained_on": "train only", "n_train": len(train), "selection": "fixed baseline specified before predictions; no tuning", "parameters": model.get_params(deep=False) if name != "LR" else store.cfg["model"]["lr"]}
    scaler = lr.named_steps["preprocess"].named_transformers_["age"]
    assert np.allclose(scaler.mean_, X[train, 0].mean())
    metadata["LR"]["age_scaler_mean"] = scaler.mean_.tolist()
    metadata["LR"]["age_scaler_scale"] = scaler.scale_.tolist()
    write_json(store.artifacts / "models/model_metadata.json", metadata)
    predictions.to_csv(store.results / "predictions_validation.csv", index=False)
    pd.DataFrame(metrics).to_csv(store.results / "T2_validation_metrics.csv", index=False)
    store.stage("E1", "complete", {"validation_metrics": metrics, "tuned": False})
    return models


def load_models(store):
    return {"LR": joblib.load(store.artifacts / "models/model_lr.joblib"), "RF_seed11": joblib.load(store.artifacts / "models/model_rf_seed11.joblib")}
