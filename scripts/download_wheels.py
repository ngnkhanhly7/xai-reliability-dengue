"""Fetch official PyPI distributions using bounded HTTP ranges on slow links."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time
import urllib.request

from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".wheelhouse"
DEST.mkdir(exist_ok=True)
TAGS = {tag: i for i, tag in enumerate(sys_tags())}

# All direct packages plus the transitive packages of this pinned environment.
PACKAGES = dict(line.strip().split("==") for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip())
PACKAGES.update({"numba": "0.61.2", "llvmlite": "0.44.0", "scikit-image": "0.25.2", "slicer": "0.0.8"})
for name in ["tqdm", "cloudpickle", "typing-extensions", "packaging", "networkx", "imageio", "pillow", "tifffile", "lazy-loader", "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing", "python-dateutil", "pytz", "tzdata", "six", "colorama", "setuptools", "wheel"]:
    PACKAGES.setdefault(name, None)


def json_url(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def distribution(name, version):
    url = f"https://pypi.org/pypi/{name}/{version}/json" if version else f"https://pypi.org/pypi/{name}/json"
    data = json_url(url)
    options = []
    for item in data["urls"]:
        if item["filename"].endswith(".whl"):
            _, _, _, tags = parse_wheel_filename(item["filename"])
            scores = [TAGS[tag] for tag in tags if tag in TAGS]
            if scores:
                options.append((min(scores), item))
    if options:
        return min(options, key=lambda x: x[0])[1]
    if name == "lime":
        return next(x for x in data["urls"] if x["packagetype"] == "sdist")
    raise RuntimeError(f"No compatible wheel for {name} {version}")


def chunk(url, start, end):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "Dengue-research-bootstrap"})
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                if len(content) != end - start + 1:
                    raise ValueError(f"Unexpected range size: {len(content)}")
                return content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(.5)


def download(name, version):
    item = distribution(name, version)
    path = DEST / item["filename"]
    if path.exists() and sha256(path.read_bytes()).hexdigest() == item["digests"]["sha256"]:
        return item["filename"]
    size, step = item["size"], 512 * 1024
    with ThreadPoolExecutor(max_workers=4) as pool:
        pieces = list(pool.map(lambda bounds: chunk(item["url"], *bounds), [(i, min(i + step, size) - 1) for i in range(0, size, step)]))
    content = b"".join(pieces)
    if sha256(content).hexdigest() != item["digests"]["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {name}")
    path.write_bytes(content)
    print(item["filename"], size, "SHA256 OK", flush=True)
    return item["filename"]


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [(name, pool.submit(download, name, version)) for name, version in PACKAGES.items()]
        for name, future in tasks:
            future.result()
