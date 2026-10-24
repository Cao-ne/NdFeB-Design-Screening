"""Check hit counts for the design space and inspect a known Nd2Fe14B entry."""
import os
import sys

import requests

BASE = "https://api.materialsproject.org/materials/summary/"
FIELDS = ("material_id,formula_pretty,chemsys,elements,total_magnetization,"
          "total_magnetization_normalized_formula_units,ordering,is_magnetic,"
          "energy_above_hull,is_stable,density,volume,nsites,symmetry")


def get(key, **params):
    params.setdefault("_limit", 1)
    params.setdefault("_fields", FIELDS)
    r = requests.get(BASE, params=params, headers={"X-API-KEY": key}, timeout=90)
    return r.json() if r.status_code == 200 else {"error": r.status_code, "body": r.text[:200]}


def main():
    key = os.environ["MP_API_KEY"]
    queries = [
        ("elements=Fe,B", {"elements": "Fe,B"}),
        ("elements=Nd,Fe", {"elements": "Nd,Fe"}),
        ("elements=Nd,Fe,B", {"elements": "Nd,Fe,B"}),
        ("chemsys=Nd-Fe-B", {"chemsys": "Nd-Fe-B"}),
        ("chemsys=Nd-Fe", {"chemsys": "Nd-Fe"}),
        ("formula=Nd2Fe14B", {"formula": "Nd2Fe14B"}),
        ("formula=Fe14Nd2B", {"formula": "Fe14Nd2B"}),
        ("elements=Nd,Fe,B,Co", {"elements": "Nd,Fe,B,Co"}),
        ("elements=Nd,Fe,B,Cu", {"elements": "Nd,Fe,B,Cu"}),
        ("elements=Nd,Fe,B,Ga", {"elements": "Nd,Fe,B,Ga"}),
        ("elements=Dy,Fe,B", {"elements": "Dy,Fe,B"}),
        ("is_magnetic=true & elements=Nd,Fe,B",
         {"elements": "Nd,Fe,B", "is_magnetic": "true"}),
        ("all magnetic Nd-containing", {"elements": "Nd", "is_magnetic": "true"}),
    ]
    for label, params in queries:
        j = get(key, **params)
        if "error" in j:
            print(f"{label:<38} ERROR {j}")
            continue
        total = j.get("meta", {}).get("total_doc")
        sample = j.get("data", [{}])[0] if j.get("data") else {}
        print(f"{label:<38} total_doc={total}")
        if sample:
            print(f"    sample: {sample.get('material_id')} {sample.get('formula_pretty')} "
                  f"mag={sample.get('total_magnetization')} "
                  f"mag/fu={sample.get('total_magnetization_normalized_formula_units')} "
                  f"ord={sample.get('ordering')} "
                  f"ehull={sample.get('energy_above_hull')}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
