"""Find which OPTIMADE response_fields and pagination style MP accepts."""
import sys

import requests

BASE = "https://optimade.materialsproject.org/v1/structures"
FILT = 'elements HAS ALL "Nd","Fe","B"'


def go(label, params):
    try:
        r = requests.get(BASE, params=params, timeout=90)
        print(f"[{label}] {r.status_code} len={len(r.text)} :: {r.text[:260]}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] ERR {type(exc).__name__}: {exc}")


def main():
    go("plain", {"filter": FILT, "page_limit": 2})
    go("offset", {"filter": FILT, "page_limit": 2, "page_offset": 0})
    go("rf-one", {"filter": FILT, "page_limit": 2, "response_fields": "chemical_formula_reduced"})
    go("rf-many", {
        "filter": FILT,
        "page_limit": 2,
        "response_fields": "chemical_formula_reduced,space_group_symbols,lattice_vectors,nelements,nsites,elements_ratios",
    })
    go("rf-bad", {"filter": FILT, "page_limit": 2, "response_fields": "elements_ratios,nsites"})
    # full attribute dump for one entry
    r = requests.get(BASE, params={"filter": FILT, "page_limit": 1}, timeout=90)
    print("\nATTR KEYS:", sorted(r.json()["data"][0]["attributes"].keys()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
