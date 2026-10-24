"""Probe AFLOW REST API for RE-TM-B compounds and inspect available fields."""
import json
import sys

import requests


def get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=90)
        return r.status_code, r.text
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def main():
    tests = [
        "https://aflow.org/API/aflux/?catalog(Fe,Nd,B),nspecies(3),$paging(0,2),$format(json)",
        "https://aflow.org/API/aflux/?catalog(Fe,Nd,B),$paging(0,2),$format(json)",
        "https://aflow.org/API/aflux/?species(Fe),species(Nd),species(B),nspecies(3),$paging(0,2),$format(json)",
        "https://aflow.org/API/aflux/?agl(Fe-Nd-B),$paging(0,2),$format(json)",
    ]
    for u in tests:
        code, text = get(u)
        print("URL:", u)
        print("CODE:", code, "LEN:", len(text))
        print("BODY:", text[:700].replace("\n", " "))
        print("-" * 70)

    # OQMD retry
    for attempt in range(2):
        code, text = get(
            "https://oqmd.org/oqmdapi/formationenergy",
            {
                "filter": "element_set=Fe-Nd-B",
                "fields": "name,spacegroup,delta_e,stability,natoms,unit_cell",
                "limit": 2,
                "format": "json",
            },
        )
        print("OQMD", attempt, code, text[:200].replace("\n", " "))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
