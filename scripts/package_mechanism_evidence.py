"""Package v3 alongside byte-preserved v2 evidence; never overwrite v1/v2 ZIPs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from package_evidence import digest, inspect_archive

ROOT = Path(__file__).resolve().parents[1]
RUN = "dengue_20261007_mechanism_v3"


def build():
    parent_path = ROOT / "release/dengue_evidence_v2.zip"
    destination = ROOT / "release/dengue_evidence_v3.zip"
    lock = json.loads((ROOT / "artifacts" / RUN / "locked_before_v3_fits.json").read_text(encoding="utf-8"))
    assert digest(parent_path.read_bytes()) == lock["archive_v2_sha256"]
    with zipfile.ZipFile(parent_path) as parent:
        old_inventory = json.loads(parent.read("EVIDENCE_MANIFEST.json"))["sha256"]
        for name, checksum in old_inventory.items():
            assert digest((ROOT / name).read_bytes()) == checksum
        fixed = ["bao-cao-quyet-dinh-phat-trien-lime-shap-dengue.md", "README_v3.md",
                 "report/research_brief_lab.md", "report/mechanism_protocol.md", "configs/mechanism.yaml",
                 "src/mechanism_analysis.py", "scripts/summarize_mechanism.py", "scripts/package_mechanism_evidence.py",
                 "release/dengue_evidence_v2.zip"]
        sources = {ROOT / name for name in fixed}
        for folder in ["artifacts", "results", "figures"]:
            sources.update(path for path in (ROOT / folder / RUN).rglob("*")
                           if path.is_file() and path.suffix not in {".tmp", ".pyc"})
        report_manifest_path = ROOT / "artifacts" / RUN / "report_materials_manifest.json"
        sources.discard(report_manifest_path)
        material = {"analysis_date_client": "2026-10-07", "prepared_at_machine_clock": datetime.now(timezone.utc).isoformat(),
                    "scope": "v3 report and descriptive summaries after frozen fits",
                    "sha256": {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in sorted(sources)}}
        report_manifest_path.write_text(json.dumps(material, ensure_ascii=False, indent=2), encoding="utf-8")
        sources.add(report_manifest_path)
        new_inventory = {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in sorted(sources)}
        assert not set(old_inventory) & set(new_inventory)
        inventory = {"analysis_date_client": "2026-10-07", "created_at": datetime.now(timezone.utc).isoformat(),
                     "scope": "v1+v2+exploratory-mechanism-v3", "previous_files_unchanged": len(old_inventory),
                     "sha256": old_inventory | new_inventory}
        temporary = destination.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in old_inventory:
                archive.writestr(name, parent.read(name))
            for path in sorted(sources):
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.writestr("EVIDENCE_MANIFEST.json", json.dumps(inventory, ensure_ascii=False, indent=2))
    temporary.replace(destination)
    result = inspect_archive(destination)
    result.update({"previous_files_unchanged": len(old_inventory), "added_files": len(new_inventory),
                   "parent_v2_archive_embedded": True, "fresh_dependency_install_tested": False})
    destination.with_suffix(".verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
