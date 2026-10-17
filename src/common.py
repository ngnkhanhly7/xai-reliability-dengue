from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "sex", "NS1", "IgM", "IgG"]


def now():
    return datetime.now(timezone(timedelta(hours=7))).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Store:
    def __init__(self, run_id, config_path=None):
        self.run_id = run_id
        self.config_path = Path(config_path) if config_path else ROOT / "configs/main.yaml"
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.config_hash = sha(self.config_path)
        self.results = ROOT / "results" / run_id
        self.artifacts = ROOT / "artifacts" / run_id
        self.figures = ROOT / "figures" / run_id
        for p in [self.results, self.artifacts, self.figures, self.results / "explanation_cache", self.artifacts / "models", self.artifacts / "splits"]:
            p.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.artifacts / "run_manifest.json"
        if self.manifest_path.exists():
            self.manifest = read_json(self.manifest_path)
            if self.manifest["config_sha256"] != self.config_hash:
                raise ValueError("Config changed: use a new run_id; do not overwrite a frozen run.")
        else:
            self.manifest = {"run_id": run_id, "started_at": now(), "timezone": "UTC+07:00", "config_sha256": self.config_hash, "stages": {}, "status": "running"}
            write_json(self.artifacts / "config_frozen.json", self.cfg)
            self.save()

    def save(self):
        write_json(self.manifest_path, self.manifest)

    def stage(self, name, status, details=None):
        self.manifest["stages"][name] = {"status": status, "timestamp": now(), "details": details or {}}
        self.save()
        print(f"[{now()}] {name}: {status} {details or ''}", flush=True)

    def check_stage(self, name):
        if self.manifest["stages"].get(name, {}).get("status") != "complete":
            raise RuntimeError(f"Stage {name} must finish before this action.")

    def cache_key(self, metadata):
        fields = [metadata[k] for k in ["context", "model_id", "explainer", "split", "row_id", "seed", "budget_requested", "background_id"]]
        return hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:24]

    def get(self, metadata):
        path = self.results / "explanation_cache" / (self.cache_key(metadata) + ".json")
        return read_json(path) if path.exists() else None

    def put(self, metadata):
        write_json(self.results / "explanation_cache" / (self.cache_key(metadata) + ".json"), metadata)

    def environment(self):
        import lime.lime_tabular
        import lime.lime_base
        import shap.explainers._kernel
        from threadpoolctl import threadpool_info
        packages = ["numpy", "pandas", "scipy", "scikit-learn", "lime", "shap", "matplotlib", "joblib", "PyYAML", "psutil", "threadpoolctl"]
        modules = [lime.lime_tabular, lime.lime_base, shap.explainers._kernel]
        info = {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
            "ram_gib": psutil.virtual_memory().total / 2**30, "rf_n_jobs": self.cfg["model"]["rf"]["n_jobs"],
            "packages": {p: importlib.metadata.version(p) for p in packages},
            "threadpools": threadpool_info(), "installed_source_hashes": {m.__name__: sha(inspect.getfile(m)) for m in modules},
            "input_documents": {p: sha(ROOT / p) for p in ["bao-cao-research-lime-shap-dengue.md", "ke-hoach-thuc-nghiem-lime-shap-dengue.md"]},
        }
        path = self.artifacts / "environment.json"
        if path.exists():
            original = read_json(path)
            for field in ["python", "packages", "installed_source_hashes", "input_documents"]:
                if original[field] != info[field]:
                    raise ValueError(f"Frozen run {field} changed: use a new run_id.")
            return original
        write_json(path, info)
        return info
