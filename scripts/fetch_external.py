"""Download the machine-readable datasets that the local papers point to.

Sources are taken from data/04_literature_data_links.csv (Zenodo + GitHub). Only small
CSV/XLSX files are pulled, and everything lands in data/external/ with an index row.

Output: data/external/*  +  data/07_external_datasets.csv
"""
import json
import os
import sys

import requests

OUT_DIR = r"C:\Users\Administrator\Desktop\大创申报\data"
EXT_DIR = os.path.join(OUT_DIR, "external")
MAX_BYTES = 30 * 1024 * 1024
WANTED_EXT = (".csv", ".xlsx", ".xls", ".json", ".txt")

ZENODO_RECORDS = ["20044097"]
GITHUB_REPOS = [
    "RunqingYang1996/MCEdatabase",
    "heisammoustafa/Micromagnetic_GNN",
    "mahata-lab/soft-magnet-ML",
    "ml-AMD/exa-amd",
]

index = []


def save(name, url, resp):
    path = os.path.join(EXT_DIR, name)
    with open(path, "wb") as fh:
        fh.write(resp.content)
    size = os.path.getsize(path)
    index.append({"dataset": name, "source_url": url, "bytes": size, "local_path": path})
    print(f"  saved {name} ({size/1024:.0f} KB)")


def do_zenodo(rec):
    url = f"https://zenodo.org/api/records/{rec}"
    try:
        meta = requests.get(url, timeout=60).json()
    except Exception as exc:  # noqa: BLE001
        print(f"ZENODO {rec}: {type(exc).__name__}")
        return
    title = meta.get("metadata", {}).get("title", "")[:90]
    print(f"ZENODO {rec}: {title}")
    for f in meta.get("files", [])[:20]:
        key = f.get("key", "file")
        if not key.lower().endswith(WANTED_EXT):
            continue
        if f.get("size", 0) > MAX_BYTES:
            print(f"  skip (too big): {key}")
            continue
        link = f.get("links", {}).get("self")
        try:
            r = requests.get(link, timeout=300)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"  fail {key}: {type(exc).__name__}")
            continue
        save(f"zenodo_{rec}_{key}", link, r)


def do_github(repo):
    api = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    try:
        r = requests.get(api, timeout=60, headers={"Accept": "application/vnd.github+json"})
        if r.status_code != 200:
            print(f"GITHUB {repo}: HTTP {r.status_code}")
            return
        tree = r.json().get("tree", [])
    except Exception as exc:  # noqa: BLE001
        print(f"GITHUB {repo}: {type(exc).__name__}")
        return
    picks = [t for t in tree
             if t.get("type") == "blob"
             and t.get("path", "").lower().endswith(WANTED_EXT)
             and t.get("size", 0) <= MAX_BYTES]
    print(f"GITHUB {repo}: {len(picks)} candidate files")
    for t in picks[:4]:
        raw = f"https://raw.githubusercontent.com/{repo}/HEAD/{t['path']}"
        try:
            r = requests.get(raw, timeout=180)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"  fail {t['path']}: {type(exc).__name__}")
            continue
        flat = t["path"].replace("/", "__")
        save(f"{repo.replace('/', '_')}__{flat}", raw, r)


def main():
    os.makedirs(EXT_DIR, exist_ok=True)
    for rec in ZENODO_RECORDS:
        do_zenodo(rec)
    for repo in GITHUB_REPOS:
        do_github(repo)

    path = os.path.join(OUT_DIR, "07_external_datasets.csv")
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "source_url", "bytes", "local_path"])
        w.writeheader()
        w.writerows(index)
    print(f"WROTE {path} rows={len(index)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
