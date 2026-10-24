"""Pull magnetic properties from the Materials Project API (needs MP_API_KEY).

Two pools:
  design_space : RE-Fe-B and Nd-Fe-B-X compounds - the compositions this project tunes
  context      : RE-Fe and Fe-B based magnetic materials - pretraining / comparison pool

Derived columns (standard CGS <-> SI conversion, clearly marked as derived):
  M_emu_cm3 = 9274 * total_magnetization / volume_A3
  Ms_T      = 4*pi*M_emu_cm3 / 1e4        (mu0*M, the quantity compared with experiment)
  Ms_emu_g  = M_emu_cm3 / density

Sanity anchor: Nd2Fe14B (mp-aaaaahri) comes out at ~1.56 T vs ~1.61 T experiment.

Output: data/09_mp_magnetism.csv
"""
import csv
import json
import math
import os
import sys
import time

import requests

BASE = "https://api.materialsproject.org/materials/summary/"
OUT_DIR = r"C:\Users\Administrator\Desktop\大创申报\data"

FIELDS = [
    "material_id", "formula_pretty", "formula_anonymous", "composition_reduced",
    "elements", "nelements", "chemsys", "symmetry", "density", "volume", "nsites",
    "is_stable", "is_magnetic", "ordering", "energy_above_hull",
    "formation_energy_per_atom", "total_magnetization",
    "total_magnetization_normalized_formula_units",
    "total_magnetization_normalized_vol", "num_magnetic_sites",
    "num_unique_magnetic_sites", "types_of_magnetic_species", "theoretical",
    "deprecated", "last_updated",
]

RE = ["Nd", "Pr", "Dy", "Tb", "Ce", "Sm", "Gd", "Y", "Ho", "Er", "La", "Lu"]
SUB = ["Co", "Cu", "Ga", "Al", "Si", "V", "Ti", "Nb", "Zr", "Mn", "Ni", "Cr"]
TRACKED = RE + SUB + ["Fe", "B"]


def build_queries():
    queries = []
    for el in RE:
        queries.append((f"design_space_RE-Fe-B::{el}", "design_space",
                        {"elements": f"{el},Fe,B"}))
    for el in SUB:
        queries.append((f"design_space_Nd-Fe-B-{el}", "design_space",
                        {"elements": f"Nd,Fe,B,{el}"}))
    for el in RE:
        queries.append((f"context_{el}-Fe", "context", {"elements": f"{el},Fe"}))
    for el in RE:
        queries.append((f"context_{el}-Co", "context", {"elements": f"{el},Co"}))
    queries.append(("context_Fe-B", "context", {"elements": "Fe,B"}))
    queries.append(("context_Fe-Co-B", "context", {"elements": "Fe,Co,B"}))
    return queries


def fetch(key, params):
    """Yield every document for a query, 1000 at a time."""
    skip = 0
    while True:
        p = dict(params)
        p["_fields"] = ",".join(FIELDS)
        p["_limit"] = 1000
        p["_skip"] = skip
        r = requests.get(BASE, params=p, headers={"X-API-KEY": key}, timeout=180)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}: {r.text[:140]}")
            return
        payload = r.json()
        data = payload.get("data", [])
        if not data:
            return
        for doc in data:
            yield doc
        total = payload.get("meta", {}).get("total_doc", 0)
        skip += len(data)
        if skip >= total:
            return
        time.sleep(0.15)


def derived(doc):
    vol = doc.get("volume")
    mag = doc.get("total_magnetization")
    rho = doc.get("density")
    out = {"M_emu_cm3": None, "Ms_T": None, "Ms_emu_g": None, "moment_uB_fu": None}
    if vol and mag is not None:
        m_vol = 9274.0 * mag / vol
        out["M_emu_cm3"] = round(m_vol, 3)
        out["Ms_T"] = round(4.0 * math.pi * m_vol / 1e4, 4)
        if rho:
            out["Ms_emu_g"] = round(m_vol / rho, 3)
    out["moment_uB_fu"] = doc.get("total_magnetization_normalized_formula_units")
    return out


def composition_columns(doc):
    """Atomic fractions per element, taken from MP's reduced composition."""
    comp = doc.get("composition_reduced") or {}
    total = sum(float(v) for v in comp.values()) or 0.0
    cols = {"composition_reduced": json.dumps(
        {k: round(float(v), 6) for k, v in comp.items()}, ensure_ascii=False)}
    if total <= 0:
        return cols
    fracs = {el: 0.0 for el in TRACKED}
    other = 0.0
    for el, v in comp.items():
        f = float(v) / total
        if el in fracs:
            fracs[el] = f
        else:
            other += f
    cols.update({f"frac_{el}": round(val, 6) for el, val in fracs.items()})
    cols["frac_other"] = round(other, 6)
    return cols


def main():
    key = os.environ.get("MP_API_KEY", "")
    if not key:
        print("MP_API_KEY not set")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = {}
    for tag, pool, params in build_queries():
        n = 0
        for doc in fetch(key, params):
            mid = doc.get("material_id")
            if not mid:
                continue
            if mid in rows:
                rows[mid]["query_tags"] += f"; {tag}"
                continue
            sym = doc.get("symmetry") or {}
            row = {
                "source": "Materials Project API (X-API-KEY)",
                "material_id": mid,
                "pool": pool,
                "query_tags": tag,
                "formula_pretty": doc.get("formula_pretty"),
                "formula_anonymous": doc.get("formula_anonymous"),
                "chemsys": doc.get("chemsys"),
                "elements": "-".join(doc.get("elements") or []),
                "n_elements": doc.get("nelements"),
                "n_sites": doc.get("nsites"),
                "crystal_system": sym.get("crystal_system"),
                "space_group_symbol": sym.get("symbol"),
                "space_group_number": sym.get("number"),
                "density_g_cm3": doc.get("density"),
                "volume_A3": doc.get("volume"),
                "is_stable": doc.get("is_stable"),
                "is_magnetic": doc.get("is_magnetic"),
                "ordering": doc.get("ordering"),
                "energy_above_hull_eV": doc.get("energy_above_hull"),
                "formation_energy_per_atom_eV": doc.get("formation_energy_per_atom"),
                "total_magnetization_uB_cell": doc.get("total_magnetization"),
                "moment_uB_per_fu": doc.get("total_magnetization_normalized_formula_units"),
                "moment_uB_per_A3": doc.get("total_magnetization_normalized_vol"),
                "n_magnetic_sites": doc.get("num_magnetic_sites"),
                "n_unique_magnetic_sites": doc.get("num_unique_magnetic_sites"),
                "magnetic_species": json.dumps(doc.get("types_of_magnetic_species")
                                               or [], ensure_ascii=False),
                "theoretical": doc.get("theoretical"),
                "deprecated": doc.get("deprecated"),
                "last_updated": doc.get("last_updated"),
                "mp_url": f"https://next-gen.materialsproject.org/materials/{mid}",
            }
            row.update(derived(doc))
            row.update(composition_columns(doc))
            rows[mid] = row
            n += 1
        print(f"  {tag:<34} fetched {n}")

    out_rows = list(rows.values())
    cols = list(out_rows[0].keys())
    path = os.path.join(OUT_DIR, "09_mp_magnetism.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"WROTE {path} rows={len(out_rows)}")

    ds = [r for r in out_rows if r["pool"] == "design_space"]
    print(f"design_space rows={len(ds)}  context rows={len(out_rows)-len(ds)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
