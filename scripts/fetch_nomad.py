"""Fetch Nd-Fe-B family entries from NOMAD (FAIR repository of computed/measured data).

Output: data/02_nomad_entries.csv
"""
import csv
import json
import os
import sys
import time

import requests

API = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"
OUT_DIR = r"C:\Users\Administrator\Desktop\大创申报\data"
RE_ELEMENTS = ["Nd", "Pr", "Dy", "Tb", "Ce", "Sm", "Gd", "Y"]

BASE_FIELDS = [
    "entry_id",
    "entry_name",
    "upload_id",
    "mainfile",
    "results.material.chemical_formula_hill",
    "results.material.chemical_formula_reduced",
    "results.material.elements",
    "results.material.n_elements",
    "results.material.material_id",
    "results.material.symmetry.space_group_symbol",
    "results.material.symmetry.crystal_system",
    "results.properties.available_properties",
    "results.method.simulation.program_name",
    "datasets",
]


def query(body):
    r = requests.post(API, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def pick_fields():
    """Keep only the quantities this NOMAD deployment accepts."""
    ok = []
    for field in BASE_FIELDS:
        body = {
            "query": {"results.material.elements:all": ["Nd", "Fe", "B"]},
            "pagination": {"page_size": 1},
            "required": {"include": [field]},
        }
        try:
            query(body)
            ok.append(field)
        except Exception:  # noqa: BLE001
            print(f"  drop field: {field}")
    return ok


def flatten(entry, fields, re_el):
    attr = entry

    def dig(path):
        cur = attr
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    row = {"rare_earth": re_el}
    for f in fields:
        val = dig(f)
        if isinstance(val, (list, dict)):
            val = json.dumps(val, ensure_ascii=False)
        row[f] = val
    eid = entry.get("entry_id")
    row["nomad_url"] = f"https://nomad-lab.eu/prod/v1/gui/entry/id/{eid}"
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fields = pick_fields()
    print("usable fields:", len(fields))

    rows = []
    seen = set()
    for re_el in RE_ELEMENTS:
        page_after = None
        n_el = 0
        while True:
            body = {
                "query": {"results.material.elements:all": [re_el, "Fe", "B"]},
                "pagination": {"page_size": 500},
                "required": {"include": fields},
            }
            if page_after:
                body["pagination"]["page_after_value"] = page_after
            try:
                payload = query(body)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {re_el} page failed: {type(exc).__name__}")
                break
            data = payload.get("data", [])
            if not data:
                break
            for entry in data:
                eid = entry.get("entry_id")
                if eid in seen:
                    continue
                seen.add(eid)
                rows.append(flatten(entry, fields, re_el))
                n_el += 1
            page_after = payload.get("pagination", {}).get("next_page_after_value")
            if not page_after:
                break
            time.sleep(0.15)
        print(f"  {re_el}+Fe+B -> {n_el} entries")

    cols = ["rare_earth"] + fields + ["nomad_url"]
    path = os.path.join(OUT_DIR, "02_nomad_entries.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {path} rows={len(rows)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
