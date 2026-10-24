"""Size up the pools we could pull: NdFeB design space vs. broader magnetic pool."""
import os
import sys

import requests

BASE = "https://api.materialsproject.org/materials/summary/"
RE = ["Nd", "Pr", "Dy", "Tb", "Ce", "Sm", "Gd", "Y", "Ho", "Er", "La", "Lu"]
SUB = ["Co", "Cu", "Ga", "Al", "Si", "V", "Ti", "Nb", "Zr", "Mn", "Ni", "Cr"]


def count(key, **params):
    params["_limit"] = 1
    params["_fields"] = "material_id"
    r = requests.get(BASE, params=params, headers={"X-API-KEY": key}, timeout=90)
    if r.status_code != 200:
        return f"ERR{r.status_code}"
    return r.json().get("meta", {}).get("total_doc")


def main():
    key = os.environ["MP_API_KEY"]
    total = 0
    print("--- design space: RE-Fe-B ---")
    for el in RE:
        c = count(key, elements=f"{el},Fe,B")
        print(f"  {el}-Fe-B: {c}")
    print("--- design space: Nd-Fe-B + X ---")
    for el in SUB:
        c = count(key, elements=f"Nd,Fe,B,{el}")
        print(f"  Nd-Fe-B-{el}: {c}")
    print("--- pretraining / context pools ---")
    pools = [
        ("Fe,B", {"elements": "Fe,B"}),
        ("Nd (all)", {"elements": "Nd"}),
        ("Pr (all)", {"elements": "Pr"}),
        ("Sm (all)", {"elements": "Sm"}),
        ("Nd,Fe (all)", {"elements": "Nd,Fe"}),
        ("Sm,Co", {"elements": "Sm,Co"}),
        ("Fe,Nd,Co", {"elements": "Fe,Nd,Co"}),
        ("Ce,Fe,B", {"elements": "Ce,Fe,B"}),
        ("Y,Fe,B", {"elements": "Y,Fe,B"}),
        ("Fe,Co,B", {"elements": "Fe,Co,B"}),
    ]
    for label, params in pools:
        print(f"  {label}: {count(key, **params)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
