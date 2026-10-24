"""Mine the local NdFeB / permanent-magnet literature (36 PDFs) into structured records.

Two outputs:
  data/03_literature_records.csv  - composition + magnetic property records with the
                                    sentence they came from (human-verifiable evidence)
  data/04_literature_data_links.csv - repository links (Zenodo/GitHub/Materials Cloud/...)
                                    found in the papers, i.e. pointers to downloadable
                                    curated datasets
  data/05_literature_tables.csv   - raw tables lifted out of the PDFs (long format),
                                    for the review/database papers that compile data
"""
import csv
import os
import re
import sys

import fitz  # PyMuPDF

PDF_DIR = r"C:\Users\Administrator\Desktop\大创申报\文献"
OUT_DIR = r"C:\Users\Administrator\Desktop\大创申报\data"

MAGNET_ELEMENTS = {
    "Nd", "Pr", "Dy", "Tb", "Ce", "Sm", "Gd", "Ho", "Er", "Y", "La", "Lu",
    "Fe", "Co", "Cu", "Ga", "Al", "V", "Ti", "Nb", "Zr", "Si", "Mn", "Ni", "Cr", "B",
}

# plain formula token, e.g. Nd2Fe14B, Fe14Nd2B, Nd12.5Fe80.6B6.9
RE_FORMULA = re.compile(r"\b(?=[A-Za-z0-9.]*\d)(?:[A-Z][a-z]?[0-9]*\.?[0-9]*?){2,}\b")
# substitution style, e.g. (Nd,Dy)2Fe14B, (Nd0.8Dy0.2)2Fe14B, Nd2Fe14-xCoxB
RE_FORMULA_SUB = re.compile(r"\([A-Za-z][A-Za-z0-9.,\-\s]*\)\s*[0-9]*\.?[0-9]*[A-Za-z0-9]*")
RE_VALUE_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(kOe|Oe|MGOe|MG\s*Oe|kA/m|kA\s*m-1|kA\s*m−1|A/m|MA/m|kJ/m3|kJ\s*m-3|kJ\s*m−3|"
    r"kJ·m-3|kG|G\b|emu/g|emu\s*g-1|am2/kg|mT|kGs|T\b|K\b|°C|C\b)"
)
RE_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9(\u4e00-\u9fff])")

PROPERTY_PATTERNS = [
    ("coercivity_Hc", r"(coercivit\w*|\bH\s*cj?\b|\bHcj\b|\biHc\b|\bHci\b|μ0Hc|µ0Hc|"
                      r"intrinsic coercivity|矫顽力)"),
    ("remanence_Br", r"(remanence|remanent\s+magnetization|\bB\s*r\b|\bBr\b|剩磁)"),
    ("energy_product_BHmax", r"(energy\s+product|\(BH\)\s*max|\(BH\)max|BHmax|maximum\s+energy"
                             r"|最大磁能积|磁能积)"),
    ("saturation_magnetization_Ms", r"(saturation\s+magnetization|\bM\s*s\b|\bMs\b|"
                                    r"饱和磁化强度|饱和磁极化强度)"),
    ("anisotropy_field_Ha", r"(anisotropy\s+field|\bH\s*a\b|\bHa\b|各向异性场)"),
    ("curie_temperature_Tc", r"(Curie\s+temperature|\bT\s*c\b|\bTc\b|居里温度)"),
]

# unit handling: (SI unit, factor to SI, which properties it may belong to)
UNIT_RULES = {
    "kOe": ("kA/m", 79.5775, {"coercivity_Hc", "anisotropy_field_Ha"}),
    "Oe": ("kA/m", 0.0795775, {"coercivity_Hc", "anisotropy_field_Ha"}),
    "kA/m": ("kA/m", 1.0, {"coercivity_Hc", "anisotropy_field_Ha"}),
    "kA m-1": ("kA/m", 1.0, {"coercivity_Hc", "anisotropy_field_Ha"}),
    "kA m−1": ("kA/m", 1.0, {"coercivity_Hc", "anisotropy_field_Ha"}),
    "A/m": ("kA/m", 0.001, {"coercivity_Hc", "anisotropy_field_Ha"}),
    "MA/m": ("kA/m", 1000.0, {"coercivity_Hc", "anisotropy_field_Ha"}),
    "T": ("T", 1.0, {"remanence_Br", "saturation_magnetization_Ms",
                     "coercivity_Hc", "anisotropy_field_Ha"}),
    "mT": ("T", 0.001, {"coercivity_Hc", "anisotropy_field_Ha", "remanence_Br"}),
    "kG": ("T", 0.1, {"remanence_Br", "saturation_magnetization_Ms",
                      "coercivity_Hc", "anisotropy_field_Ha"}),
    "kGs": ("T", 0.1, {"remanence_Br", "saturation_magnetization_Ms",
                       "coercivity_Hc", "anisotropy_field_Ha"}),
    "G": ("T", 0.0001, {"remanence_Br", "saturation_magnetization_Ms"}),
    "emu/g": ("emu/g", 1.0, {"saturation_magnetization_Ms", "remanence_Br"}),
    "emu g-1": ("emu/g", 1.0, {"saturation_magnetization_Ms", "remanence_Br"}),
    "am2/kg": ("emu/g", 1.0, {"saturation_magnetization_Ms", "remanence_Br"}),
    "MGOe": ("kJ/m3", 7.9577, {"energy_product_BHmax"}),
    "MG Oe": ("kJ/m3", 7.9577, {"energy_product_BHmax"}),
    "kJ/m3": ("kJ/m3", 1.0, {"energy_product_BHmax"}),
    "kJ m-3": ("kJ/m3", 1.0, {"energy_product_BHmax"}),
    "kJ m−3": ("kJ/m3", 1.0, {"energy_product_BHmax"}),
    "kJ·m-3": ("kJ/m3", 1.0, {"energy_product_BHmax"}),
    "K": ("K", 1.0, {"curie_temperature_Tc"}),
    "°C": ("K", None, {"curie_temperature_Tc"}),
    "C": ("K", None, {"curie_temperature_Tc"}),
}

# plausibility windows (SI units) - anything outside is a parsing artefact
PLAUSIBLE = {
    ("curie_temperature_Tc", "K"): (1.0, 1500.0),
    ("energy_product_BHmax", "kJ/m3"): (1.0, 600.0),
    ("saturation_magnetization_Ms", "T"): (0.05, 2.6),
    ("saturation_magnetization_Ms", "emu/g"): (1.0, 300.0),
    ("remanence_Br", "T"): (0.05, 2.0),
    ("remanence_Br", "emu/g"): (1.0, 300.0),
    ("coercivity_Hc", "kA/m"): (0.1, 4000.0),
    ("coercivity_Hc", "T"): (0.001, 6.0),
    ("anisotropy_field_Ha", "kA/m"): (1.0, 20000.0),
    ("anisotropy_field_Ha", "T"): (0.01, 30.0),
}

PROCESS_HINTS = [
    "anneal", "sinter", "grain boundary diffusion", "GBD", "hot deform",
    "hydrogenation", "HDDR", "melt spinn", "strip cast", "退火", "烧结", "晶界扩散",
    "快淬", "回火", "temper",
]


def clean_formula(tok):
    tok = tok.strip().strip(".,;:-")
    if not re.search(r"[A-Z][a-z]?", tok):
        return None
    elems = re.findall(r"[A-Z][a-z]?", tok)
    if len(set(elems) & MAGNET_ELEMENTS) < 2:
        return None
    if not ({"Fe"} & set(elems) or {"Nd", "Pr", "Dy", "Tb", "Sm", "Ce"} & set(elems)):
        return None
    if len(tok) > 40:
        return None
    return tok


def nearest_composition(sentence, pos):
    """Composition token closest to the reported value - usually the one it describes."""
    cands = []
    for m in RE_FORMULA.finditer(sentence):
        c = clean_formula(m.group(0))
        if c:
            cands.append((abs(m.start() - pos), c))
    for m in RE_FORMULA_SUB.finditer(sentence):
        c = clean_formula(m.group(0))
        if c:
            cands.append((abs(m.start() - pos), c))
    if not cands:
        return ""
    cands.sort(key=lambda t: t[0])
    return cands[0][1]


def plausible(prop, si_unit, value):
    window = PLAUSIBLE.get((prop, si_unit))
    if not window:
        return True
    return window[0] <= value <= window[1]


def find_compositions(sentence):
    found = []
    for m in RE_FORMULA.finditer(sentence):
        c = clean_formula(m.group(0))
        if c:
            found.append(c)
    for m in RE_FORMULA_SUB.finditer(sentence):
        c = clean_formula(m.group(0))
        if c:
            found.append(c)
    return list(dict.fromkeys(found))


def normalize(text):
    """Undo two-column PDF damage: join hyphenated line breaks, collapse whitespace."""
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def sentences_of(text):
    out = []
    for part in RE_SENTENCE_SPLIT.split(text):
        part = part.strip()
        if 20 <= len(part) <= 600:
            out.append(part)
    return out


def process_hint(sentence):
    low = sentence.lower()
    hits = [h for h in PROCESS_HINTS if h.lower() in low]
    return "; ".join(dict.fromkeys(hits))[:120] if hits else ""


def main():
    do_tables = "--tables" in sys.argv or not os.path.exists(
        os.path.join(OUT_DIR, "05_literature_tables.csv"))
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    records = []
    links = []
    tables = []
    seen_link = set()

    for fname in files:
        path = os.path.join(PDF_DIR, fname)
        try:
            doc = fitz.open(path)
        except Exception as exc:  # noqa: BLE001
            print("SKIP", fname, type(exc).__name__)
            continue
        text_all = []
        for page in doc:
            text_all.append(page.get_text())
        doc.close()
        full = normalize("\n".join(text_all))

        # repository links
        for m in re.finditer(r"https?://[^\s\)\]\}>\"']+", full):
            url = m.group(0).rstrip(".,;")
            if not re.search(r"zenodo|github|figshare|materialscloud|nomad|10\.5281|"
                             r"dryad|dataverse|ldr\.lanl|cds\.cnrs", url):
                continue
            if url in seen_link:
                continue
            seen_link.add(url)
            start = max(0, m.start() - 120)
            links.append({
                "source_pdf": fname,
                "url": url,
                "context": re.sub(r"\s+", " ", full[start:m.end() + 60])[:260],
            })

        for sent in sentences_of(full):
            comps = find_compositions(sent)
            if not comps:
                continue
            hint = process_hint(sent)
            for prop, pat in PROPERTY_PATTERNS:
                if not re.search(pat, sent, flags=re.IGNORECASE):
                    continue
                for vm in RE_VALUE_UNIT.finditer(sent):
                    raw_val, raw_unit = vm.group(1), vm.group(2)
                    unit_key = re.sub(r"\s+", " ", raw_unit.strip())
                    rule = UNIT_RULES.get(unit_key)
                    if not rule:
                        continue
                    si_unit, factor, allowed = rule
                    if prop not in allowed:
                        continue
                    try:
                        val = float(raw_val)
                    except ValueError:
                        continue
                    if unit_key in {"°C", "C"} and prop == "curie_temperature_Tc":
                        si_val = round(val + 273.15, 2)
                    else:
                        si_val = round(val * factor, 4)
                    if not plausible(prop, si_unit, si_val):
                        continue
                    near = nearest_composition(sent, vm.start())
                    if not near:
                        continue
                    gap = abs(sent.find(near) - vm.start())
                    records.append({
                        "source_pdf": fname,
                        "property": prop,
                        "composition_nearest": near,
                        "composition_raw": " | ".join(comps),
                        "value_raw": raw_val,
                        "unit_raw": unit_key,
                        "value_SI": si_val,
                        "unit_SI": si_unit,
                        "confidence": "high" if gap <= 80 else "medium",
                        "process_hint": hint,
                        "evidence": re.sub(r"\s+", " ", sent)[:400],
                    })
                    # keep scanning: a sentence such as "Hc = 12 kOe and Br = 1.4 T"
                    # carries two records

        # tables: review / database papers compile hundreds of rows each.
        # This pass is slow, so it is cached in 05_literature_tables.csv.
        if do_tables:
            try:
                doc = fitz.open(path)
            except Exception:  # noqa: BLE001
                continue
            for pno, page in enumerate(doc, start=1):
                try:
                    found = page.find_tables()
                except Exception:  # noqa: BLE001
                    continue
                for tno, table in enumerate(getattr(found, "tables", []) or [], start=1):
                    try:
                        rows = table.extract()
                    except Exception:  # noqa: BLE001
                        continue
                    if not rows or len(rows) < 3:
                        continue
                    for rno, row in enumerate(rows):
                        cells = [re.sub(r"\s+", " ", str(c)).strip() if c is not None else ""
                                 for c in row]
                        if not any(cells):
                            continue
                        tables.append({
                            "source_pdf": fname,
                            "page": pno,
                            "table_index": tno,
                            "row_index": rno,
                            "n_cols": len(cells),
                            "cells": " | ".join(cells)[:1000],
                        })
            doc.close()

    rec_path = os.path.join(OUT_DIR, "03_literature_records.csv")
    with open(rec_path, "w", newline="", encoding="utf-8-sig") as fh:
        cols = ["source_pdf", "property", "composition_nearest", "composition_raw",
                "value_raw", "unit_raw", "value_SI", "unit_SI", "confidence",
                "process_hint", "evidence"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(records)
    print(f"WROTE {rec_path} rows={len(records)}")

    link_path = os.path.join(OUT_DIR, "04_literature_data_links.csv")
    with open(link_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["source_pdf", "url", "context"])
        w.writeheader()
        w.writerows(links)
    print(f"WROTE {link_path} rows={len(links)}")

    tab_path = os.path.join(OUT_DIR, "05_literature_tables.csv")
    if do_tables:
        with open(tab_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["source_pdf", "page", "table_index",
                                               "row_index", "n_cols", "cells"])
            w.writeheader()
            w.writerows(tables)
        print(f"WROTE {tab_path} rows={len(tables)}")
    else:
        print(f"KEPT existing {tab_path} (use --tables to refresh)")

    from collections import Counter
    print("properties:", Counter(r["property"] for r in records))
    print("pdfs with records:",
          len({r["source_pdf"] for r in records}), "of", len(files))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
