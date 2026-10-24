import json
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")
r = requests.get(
    "https://optimade.materialsproject.org/v1/structures",
    params={
        "filter": 'chemical_formula_reduced="BFe14Nd2"',
        "page_limit": 2,
        "response_fields": (
            "chemical_formula_reduced,space_group_symbol_hermann_mauguin,"
            "space_group_it_number,lattice_vectors"
        ),
    },
    timeout=60,
)
print(r.status_code)
print(json.dumps(r.json()["data"], ensure_ascii=False)[:900])
