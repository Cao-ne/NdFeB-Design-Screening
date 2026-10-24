"""Fetch Nd-Fe-B family structures from the Materials Project OPTIMADE endpoint.

No API key is required for the OPTIMADE /structures route. We pull every entry whose
element set contains a rare earth together with Fe and B - the design space of
Nd2Fe14B-type permanent magnets - plus industrially relevant substitutions
(Co/Cu/Ga/Al/V/Ti/Nb/Zr/Si/Mn/Ni/Cr).

Per entry we keep the composition, crystal-structure descriptors and the DFT
thermodynamics (formation energy per atom, energy above hull).

Output: data/01_dft_structures_mp.csv
"""
import csv
import json
import os
import sys
import time

import requests

BASE = "https://optimade.materialsproject.org/v1/structures"
OUT_DIR = r"C:\Users\Administrator\Desktop\大创申报\data"

RE_ELEMENTS = ["Nd", "Pr", "Dy", "Tb", "Ce", "Sm", "Gd", "Ho", "Er", "Y", "La", "Lu"]
SUB_ELEMENTS = ["Co", "Cu", "Ga", "Al", "V", "Ti", "Nb", "Zr", "Si", "Mn", "Ni", "Cr"]

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

# elements that get their own composition column
COLUMN_ELEMENTS = ["Nd", "Pr", "Dy", "Tb", "Ce", "Sm", "Gd", "Y", "La", "Ho", "Er", "Lu",
                   "Fe", "Co", "Cu", "Ga", "Al", "V", "Ti", "Nb", "Zr", "Si", "Mn", "Ni",
                   "Cr", "B"]


def query(filter_str, limit=100, offset=0):
    params = {
        "filter": filter_str,
        "page_limit": limit,
        "page_offset": offset,
        "response_fields": ",".join(FIELDS),
    }
    r = requests.get(BASE, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def fraction_columns(attr):
    elems = attr.get("elements") or []
    ratios = attr.get("elements_ratios") or []
    table = dict(zip(elems, ratios))
    return {f"frac_{el}": round(table.get(el, 0.0), 6) for el in COLUMN_ELEMENTS}


def stability_columns(attr):
    """Keep the GGA/GGA+U thermodynamics, which is the usual reference for RE-TM alloys."""
    stab = attr.get("_mp_stability") or {}
    chosen = stab.get("gga_gga+u") or (next(iter(stab.values())) if stab else {})
    return {
        "formation_energy_per_atom_eV": chosen.get("formation_energy_per_atom"),
        "energy_above_hull_eV": chosen.get("energy_above_hull"),
        "thermo_id": chosen.get("thermo_id"),
        "n_functional_variants": len(stab),
    }


def crystal_system_from_cell(a, b, c, al, be, ga):
    """Derived crystal system from cell metrics (MP's public route omits the space group).
    Labelled as derived so it is never mistaken for an experimental symmetry assignment."""
    if None in (a, b, c, al, be, ga):
        return None

    def eq(x, y, tol=0.02):
        return abs(x - y) <= tol * max(abs(x), abs(y))

    def is90(x):
        return abs(x - 90.0) <= 1.0

    angles90 = [is90(al), is90(be), is90(ga)]
    if all(angles90):
        if eq(a, b) and eq(b, c):
            return "cubic"
        pairs = sum([eq(a, b), eq(b, c), eq(a, c)])
        return "tetragonal" if pairs == 1 else "orthorhombic"
    if is90(al) and is90(be) and abs(ga - 120.0) <= 1.0 and eq(a, b):
        return "hexagonal"
    if eq(al, be) and eq(be, ga):
        return "rhombohedral"
    if sum(angles90) == 2:
        return "monoclinic"
    return "triclinic"


def cell_params(lattice_vectors):
    """Convert 3x3 lattice vectors to a, b, c, alpha, beta, gamma."""
    import math

    if not lattice_vectors or len(lattice_vectors) != 3:
        return (None, None, None, None, None, None)
    a_vec, b_vec, c_vec = lattice_vectors

    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    def angle(u, v):
        nu, nv = norm(u), norm(v)
        if nu == 0 or nv == 0:
            return None
        cos = sum(x * y for x, y in zip(u, v)) / (nu * nv)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    return (
        round(norm(a_vec), 4),
        round(norm(b_vec), 4),
        round(norm(c_vec), 4),
        round(angle(b_vec, c_vec), 3),
        round(angle(a_vec, c_vec), 3),
        round(angle(a_vec, b_vec), 3),
    )


def fetch_filter(filt, tag, seen, rows, cap=4000):
    offset = 0
    n_el = 0
    while n_el < cap:
        try:
            payload = query(filt, limit=100, offset=offset)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {tag} offset={offset} failed: {type(exc).__name__}")
            return n_el
        data = payload.get("data", [])
        if not data:
            return n_el
        for entry in data:
            eid = entry.get("id")
            if eid in seen:
                continue
            seen.add(eid)
            attr = entry.get("attributes", {})
            a, b, c, al, be, ga = cell_params(attr.get("lattice_vectors"))
            row = {
                "source": "Materials Project via OPTIMADE (no-key route)",
                "mp_id": eid,
                "query_tag": tag,
                "formula_reduced": attr.get("chemical_formula_reduced"),
                "formula_hill": attr.get("chemical_formula_hill"),
                "chemical_system": attr.get("_mp_chemical_system"),
                "elements": "-".join(attr.get("elements", []) or []),
                "n_elements": attr.get("nelements"),
                "n_sites": attr.get("nsites"),
                "space_group_symbol": attr.get("space_group_symbol_hermann_mauguin"),
                "space_group_number": attr.get("space_group_it_number"),
                "crystal_system_from_cell": crystal_system_from_cell(a, b, c, al, be, ga),
                "space_group_source": "MP OPTIMADE does not expose symmetry for these ids",
                "a_A": a, "b_A": b, "c_A": c,
                "alpha_deg": al, "beta_deg": be, "gamma_deg": ga,
                "volume_A3": round(a * b * c * (1 if not (al and be and ga) else 1), 6) if a else None,
                "mp_url": f"https://materialsproject.org/materials/{eid}",
            }
            row.update(fraction_columns(attr))
            row.update(stability_columns(attr))
            rows.append(row)
            n_el += 1
        if not payload.get("meta", {}).get("more_data_available", False):
            return n_el
        offset += 100
        time.sleep(0.2)
    print(f"  (cap reached for {tag})")
    return n_el


def collect():
    rows = []
    seen = set()
    stats = {}
    for re_el in RE_ELEMENTS:
        n = fetch_filter(f'elements HAS ALL "{re_el}","Fe","B"', f"{re_el}-Fe-B", seen, rows)
        stats[f"{re_el}-Fe-B"] = n
        print(f"  {re_el}+Fe+B -> {n} structures")
    for sub_el in SUB_ELEMENTS:
        filt = f'elements HAS ALL "Nd","Fe","B","{sub_el}"'
        n = fetch_filter(filt, f"Nd-Fe-B-{sub_el}", seen, rows)
        stats[f"Nd-Fe-B-{sub_el}"] = n
        print(f"  Nd+Fe+B+{sub_el} -> {n} structures")
    return rows, stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, stats = collect()
    path = os.path.join(OUT_DIR, "01_dft_structures_mp.csv")
    cols = [
        "source", "mp_id", "query_tag", "formula_reduced", "formula_hill", "chemical_system",
        "elements", "n_elements", "n_sites",
        "space_group_symbol", "space_group_number",
        "crystal_system_from_cell", "space_group_source",
        "a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg", "volume_A3",
        "formation_energy_per_atom_eV", "energy_above_hull_eV", "thermo_id",
        "n_functional_variants", "mp_url",
    ] + [f"frac_{el}" for el in COLUMN_ELEMENTS]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {path} rows={len(rows)}")
    print("per-RE counts:", json.dumps(stats))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
