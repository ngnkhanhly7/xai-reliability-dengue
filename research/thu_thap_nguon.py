"""Download primary sources and pin GitHub snapshots; standard library only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "nguon"
SOURCES.mkdir(exist_ok=True)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Dengue-literature-research/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "div", "section", "h1", "h2", "h3", "h4", "li", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def download(item):
    name, url = item
    content = fetch(url)
    (SOURCES / name).write_bytes(content)
    if name.endswith(".html"):
        parser = PlainText()
        parser.feed(content.decode("utf-8", errors="replace"))
        lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
        (SOURCES / (name[:-5] + ".txt")).write_text("\n".join(x for x in lines if x), encoding="utf-8")
    return {"file": name, "url": url, "bytes": len(content), "sha256": sha256(content).hexdigest()}


def main():
    snapshots = {}
    for repo in ["joehuang1980/DVpatient", "marcotcr/lime", "shap/shap"]:
        info = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits?per_page=1"))
        snapshots[repo] = info[0]["sha"]
    dv = snapshots["joehuang1980/DVpatient"]
    lime = snapshots["marcotcr/lime"]
    shap = snapshots["shap/shap"]
    urls = [
        ("P1_huang_2020.html", "https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008960"),
        ("P2_lime_2016.html", "https://arxiv.org/html/1602.04938"),
        ("P3_shap_2017.html", "https://arxiv.org/html/1705.07874"),
        ("P4_aas.html", "https://arxiv.org/html/1903.10464"),
        ("P5_adebayo_2022.html", "https://arxiv.org/html/2212.04629"),
        ("P5_iclr_2022.pdf", "https://openreview.net/pdf?id=xNOVfCCvDpM"),
        ("lime_api.html", "https://lime-ml.readthedocs.io/en/latest/lime.html"),
        ("shap_kernel_api.html", "https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html"),
        ("dengue_data_Final.csv", f"https://raw.githubusercontent.com/joehuang1980/DVpatient/{dv}/dengue%20data-Final.csv"),
        ("DV_verification.ipynb", f"https://raw.githubusercontent.com/joehuang1980/DVpatient/{dv}/DV%20severity%20models_verification-20200720.ipynb"),
        ("lime_tabular.py", f"https://raw.githubusercontent.com/marcotcr/lime/{lime}/lime/lime_tabular.py"),
        ("lime_base.py", f"https://raw.githubusercontent.com/marcotcr/lime/{lime}/lime/lime_base.py"),
        ("lime_discretize.py", f"https://raw.githubusercontent.com/marcotcr/lime/{lime}/lime/discretize.py"),
        ("shap_kernel.py", f"https://raw.githubusercontent.com/shap/shap/{shap}/shap/explainers/_kernel.py"),
    ]
    records = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [(name, pool.submit(download, (name, url))) for name, url in urls]
        for name, future in futures:
            try:
                record = future.result()
                print(name, record["bytes"])
                records.append(record)
            except Exception as exc:
                print(name, "FAILED", str(exc))
                records.append({"file": name, "error": str(exc)})
    manifest = {"client_research_date": "2026-10-06", "retrieved_utc": datetime.now(timezone.utc).isoformat(), "github_commits": snapshots, "sources": records}
    (ROOT / "nguon_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
