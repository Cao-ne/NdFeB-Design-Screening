"""Probe alternative free sources: ODBX (OPTIMADE index), NEMAD, Materials Cloud."""
import sys

import requests


def get(url, **kw):
    try:
        r = requests.get(url, timeout=60, **kw)
        return r.status_code, r.text[:500].replace("\n", " ")
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def main():
    tests = {
        "odbx-info": "https://optimade.odbx.science/v1/info",
        "odbx-nfeb": (
            "https://optimade.odbx.science/v1/structures"
            "?filter=elements HAS ALL \"Nd\",\"Fe\",\"B\"&page_limit=2"
        ),
        "odbx-nfeb-count": (
            "https://optimade.odbx.science/v1/structures"
            "?filter=elements HAS ALL \"Nd\",\"Fe\",\"B\"&page_limit=1"
            "&response_fields=chemical_formula_reduced"
        ),
        "optimade-mp": "https://optimade.materialsproject.org/v1/info",
        "optimade-alexandria": "https://optimade.alexandria.icams.rub.de/v1/info",
        "materialscloud": "https://optimade.materialscloud.org/v1/info",
        "nemad": "https://nemad.cheme.cmu.edu/",
        "oqmd-root": "https://oqmd.org/",
        "magndata": "https://www.cryst.ehu.es/magndata/",
    }
    for name, url in tests.items():
        code, body = get(url)
        print(f"[{name}] {code} :: {body[:260]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
