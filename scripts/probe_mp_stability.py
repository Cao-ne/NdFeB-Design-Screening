"""Inspect MP OPTIMADE '_mp_stability' payload and validate field names for bulk pull."""
import json
import sys

import requests

BASE = "https://optimade.materialsproject.org/v1/structures"
FILT = 'elements HAS ALL "Nd","Fe","B"'

FIELDS = [
    "chemical_formula_reduced",
    "chemical_formula_hill",
    "chemical_formula_anonymous",
    "elements",
    "elements_ratios",
    "nelements",
    "nsites",
    "space_group_symbol_hermann_mauguin",
    "space_group_it_number",
    "lattice_vectors",
    "dimension_types",
    "nperiodic_dimensions",
    "structure_features",
    "_mp_stability",
    "_mp_chemical_system",
]


def main():
    r = requests.get(BASE, params={"filter": FILT, "page_limit": 2}, timeout=90)
    entry = r.json()["data"][0]
    print("STABILITY SAMPLE:", json.dumps(entry["attributes"].get("_mp_stability"),
                                          ensure_ascii=False, indent=1)[:1200])

    # validate each candidate field name
    for f in FIELDS:
        rr = requests.get(BASE, params={"filter": FILT, "page_limit": 1,
                                        "response_fields": f}, timeout=90)
        print(f"{rr.status_code}  {f}")

    rr = requests.get(BASE, params={"filter": FILT, "page_limit": 1,
                                    "response_fields": ",".join(FIELDS)}, timeout=90)
    print("COMBO", rr.status_code, rr.text[:300])
    if rr.status_code == 200:
        j = rr.json()
        print("meta:", json.dumps(j["meta"], ensure_ascii=False)[:400])
        print("row:", json.dumps(j["data"][0], ensure_ascii=False)[:900])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
