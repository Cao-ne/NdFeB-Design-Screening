"""Probe NOMAD API: which quantities are accepted and what the data looks like."""
import json
import sys
import requests

API = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"

CANDIDATE_FIELDS = [
    "entry_id",
    "results.material.chemical_formula_hill",
    "results.material.chemical_formula_reduced",
    "results.material.elements",
    "results.material.n_elements",
    "results.material.material_id",
    "results.material.symmetry.space_group_symbol",
    "results.material.symmetry.crystal_system",
    "results.material.topology",
    "results.properties.structural",
    "results.properties.magnetic",
    "results.properties.thermodynamic",
    "results.properties.electronic",
    "results.properties.available_properties",
    "results.method.simulation.program_name",
    "results.method.simulation.dft.basis_set_type",
    "results.encyclopedia",
    "upload_id",
    "datasets",
    "comment",
    "entry_name",
]


def try_fields(fields):
    body = {
        "query": {"results.material.elements:all": ["Nd", "Fe", "B"]},
        "pagination": {"page_size": 1},
        "required": {"include": fields},
    }
    r = requests.post(API, json=body, timeout=90)
    return r.status_code, r.text


def main():
    ok = []
    bad = []
    # test field availability one batch at a time
    for i in range(0, len(CANDIDATE_FIELDS), 4):
        batch = CANDIDATE_FIELDS[i:i + 4]
        code, text = try_fields(batch)
        if code == 200:
            ok.extend(batch)
            print("ACCEPT", batch)
        else:
            for f in batch:
                c2, t2 = try_fields([f])
                if c2 == 200:
                    ok.append(f)
                    print("ACCEPT", f)
                else:
                    bad.append(f)
                    print("REJECT", f, t2[:160])

    print("\nACCEPTED:", json.dumps(ok, ensure_ascii=False))
    print("REJECTED:", json.dumps(bad, ensure_ascii=False))

    code, text = try_fields(ok)
    print("\nSAMPLE STATUS", code)
    if code == 200:
        data = json.loads(text)
        print("total_hits:", data.get("pagination", {}).get("total"))
        print(json.dumps(data.get("data", [])[:1], ensure_ascii=False, indent=1)[:4000])
    else:
        print(text[:800])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
