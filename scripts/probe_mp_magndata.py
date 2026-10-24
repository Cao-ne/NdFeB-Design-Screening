"""Check whether MP's OPTIMADE endpoint serves Nd-Fe-B structures without a key,
and whether MAGNDATA is reachable over plain HTTP."""
import sys

import requests
import urllib3

urllib3.disable_warnings()


def get(url, verify=True):
    try:
        r = requests.get(url, timeout=60, verify=verify)
        return r.status_code, r.text[:400].replace("\n", " ")
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def main():
    tests = [
        ("mp-optimade-elements",
         "https://optimade.materialsproject.org/v1/structures"
         "?filter=elements%20HAS%20ALL%20%22Nd%22,%22Fe%22,%22B%22&page_limit=2"),
        ("mp-optimade-formula",
         "https://optimade.materialsproject.org/v1/structures"
         "?filter=chemical_formula_reduced%3D%22Nd2Fe14B%22&page_limit=3"),
        ("magndata-http", "http://www.cryst.ehu.es/magndata/"),
        ("magndata-https-noverify", "https://www.cryst.ehu.es/magndata/"),
    ]
    for name, url in tests:
        verify = name != "magndata-https-noverify"
        code, body = get(url, verify=verify)
        print(f"[{name}] {code} :: {body[:300]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
