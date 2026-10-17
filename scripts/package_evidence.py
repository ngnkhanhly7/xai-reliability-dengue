"""Build a portable evidence archive; inspect hashes and Markdown links in the ZIP."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
V1 = "dengue_20261006_v1"
V2 = "dengue_20261006_supp_v2"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def paths(supplementary=False):
    items = ["README.md", "requirements.txt", "requirements-lock.txt", "configs", "src", "scripts",
             "data", "research", "report", "ke-hoach-research-lime-shap-dengue.md",
             "bao-cao-research-lime-shap-dengue.md", "ke-hoach-thuc-nghiem-lime-shap-dengue.md",
             "bao-cao-thuc-nghiem-lime-shap-dengue.md"]
    for run in [V1] + ([V2] if supplementary else []):
        items.extend(f"{folder}/{run}" for folder in ["artifacts", "results", "figures"])
    if supplementary:
        items += ["bao-cao-bo-sung-lime-shap-dengue-v2.md", "bao-cao-tong-hop-lime-shap-dengue.md", "release/dengue_v1_original.zip"]
    files = set()
    for item in items:
        path = ROOT / item
        if not path.exists():
            raise FileNotFoundError(path)
        files.update([path] if path.is_file() else (p for p in path.rglob("*") if p.is_file()))
    return sorted(p for p in files if "__pycache__" not in p.parts and p.suffix not in {".pyc", ".tmp"})


def inspect_archive(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        inventory = json.loads(archive.read("EVIDENCE_MANIFEST.json"))
        names = set(archive.namelist())
        for name, checksum in inventory["sha256"].items():
            assert name in names and digest(archive.read(name)) == checksum, name
        missing = []
        for name in names:
            if not name.endswith(".md"):
                continue
            document = archive.read(name).decode("utf-8-sig")
            for target in re.findall(r"\]\(([^)]+)\)", document):
                if target.startswith(("http:", "https:", "#", "app:")):
                    continue
                target = target.split("#", 1)[0].split(" \"", 1)[0]
                combined = PurePosixPath(name).parent / target
                # Resolve '..' without touching the filesystem.
                parts = []
                for part in combined.parts:
                    if part == "..":
                        parts.pop()
                    elif part not in {".", ""}:
                        parts.append(part)
                normalized = "/".join(parts)
                if normalized not in names and not any(n.startswith(normalized.rstrip("/") + "/") for n in names):
                    missing.append({"document": name, "target": target})
        assert not missing, missing
    return {"archive": str(path.relative_to(ROOT)), "files": len(inventory["sha256"]),
            "size_bytes": path.stat().st_size, "sha256": digest(path.read_bytes()),
            "crc_passed": True, "file_hashes_passed": True, "markdown_links_passed": True}


def build(supplementary=False):
    destination = ROOT / "release" / ("dengue_evidence_v2.zip" if supplementary else "dengue_v1_original.zip")
    destination.parent.mkdir(exist_ok=True)
    if not supplementary and destination.exists():
        return inspect_archive(destination)
    sources = paths(supplementary)
    inventory = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "v1+supplementary-v2" if supplementary else "original-v1-before-supplementary-edits",
                 "sha256": {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in sources}}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("EVIDENCE_MANIFEST.json", json.dumps(inventory, ensure_ascii=False, indent=2))
    result = inspect_archive(destination)
    (destination.with_suffix(".verification.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplementary", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.supplementary), indent=2))
