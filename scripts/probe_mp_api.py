"""Probe the Materials Project REST API (v2) with the user's key.

Checks: which magnetic fields exist, and whether `elements=` means "contains all".
The key is read from the MP_API_KEY environment variable and is never printed.
"""
import json
import os
import sys

import requests

BASE = "https://api.materialsproject.org/materials/summary/"

CANDIDATE_FIELDS = [
    "material_id", "formula_pretty", "formula_anonymous", "composition",
    "composition_reduced", "elements", "nelements", "chemsys", "symmetry",
    "density", "volume", "nsites", "is_stable", "is_magnetic", "ordering",
    "energy_above_hull", "formation_energy_per_atom", "band_gap",
    "total_magnetization", "total_magnetization_normalized_vol",
    "total_magnetization_normalized_formula_units",
    "num_magnetic_sites", "num_unique_magnetic_sites", "types_of_magnetic_species",
    "weighted_surface_energy_EV_PER_ANG2", "theoretical", "database_IDs",
    "origins", "last_updated", "deprecated",
]


def get(key, params):
    r = requests.get(BASE, params=params, headers={"X-API-KEY": key}, timeout=90)
    return r.status_code, r.text


def main():
    key = os.environ.get("MP_API_KEY", "")
    if not key:
        print("MP_API_KEY not set")
        return

    code, text = get(key, {"_limit": 1, "elements": "Nd,Fe,B"})
    print("baseline:", code, text[:200].replace("\n", " "))

    ok, bad = [], []
    for field in CANDIDATE_FIELDS:
        c, t = get(key, {"_limit": 1, "elements": "Nd,Fe,B", "_fields": field})
        if c == 200:
            ok.append(field)
        else:
            bad.append((field, t[:110]))
    print("\nACCEPTED FIELDS:")
    print(json.dumps(ok, ensure_ascii=False))
    print("\nREJECTED:")
    for f, t in bad:
        print(" ", f, "|", t)

    # semantics of elements= and chemsys=
    for params in (
        {"_limit": 3, "_fields": "material_id,formula_pretty,elements,chemsys",
         "elements": "Nd,Fe,B"},
        {"_limit": 3, "_fields": "material_id,formula_pretty,elements,chemsys",
         "chemsys": "Nd-Fe-B"},
        {"_limit": 3, "_fields": "material_id,formula_pretty,elements,chemsys",
         "elements": "Nd,Fe,B,Co"},
    ):
        c, t = get(key, params)
        print("\nPARAMS", {k: v for k, v in params.items() if k != "_fields"})
        print(c, t[:320].replace("\n", " "))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
