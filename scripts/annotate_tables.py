"""Flag the PDF tables that actually carry magnetic data, so 05_literature_tables.csv
can be filtered down to the ~dozen tables worth structuring by hand.

Rewrites data/05_literature_tables.csv with two extra columns.
"""
import csv
import os
import re
import sys

import pandas as pd

OUT_DIR = r"C:\Users\Administrator\Desktop\大创申报\data"
PATH = os.path.join(OUT_DIR, "05_literature_tables.csv")

UNIT_HINTS = {
    "kOe": "coercivity/field(kOe)",
    "kA/m": "coercivity/field(kA/m)",
    "kA m-1": "coercivity/field(kA/m)",
    "MA/m": "coercivity/field(MA/m)",
    "MGOe": "energy product(MGOe)",
    "kJ/m3": "energy product(kJ/m3)",
    "kJ m-3": "energy product(kJ/m3)",
    "emu/g": "magnetization(emu/g)",
    "am2/kg": "magnetization(Am2/kg)",
    "kG": "flux density(kG)",
    "T]": "flux density(T)",
    "K]": "temperature(K)",
    "°C": "temperature(C)",
}

PROP_WORDS = ["coerciv", "矫顽", "remanen", "剩磁", "energy product", "磁能积",
              "saturation magnetization", "饱和磁化", "curie", "居里",
              "anisotropy field", "各向异性场", "magnetization", "磁化强度"]


def main():
    df = pd.read_csv(PATH)
    df["cells"] = df["cells"].fillna("")

    def units_in(text):
        low = text
        hits = [label for token, label in UNIT_HINTS.items() if token in low]
        return "; ".join(dict.fromkeys(hits))[:200]

    df["magnetic_units"] = df["cells"].map(units_in)
    df["has_property_word"] = df["cells"].map(
        lambda t: "; ".join(w for w in PROP_WORDS if w.lower() in t.lower())[:160])
    df["is_magnetic_table"] = (
        (df["magnetic_units"] != "") | (df["has_property_word"] != "")
    )
    df["mentions_Nd_Fe"] = df["cells"].map(
        lambda t: bool(re.search(r"\bNd\b", t) and re.search(r"\bFe\b", t)))
    df["mentions_NdFeB"] = df["cells"].map(
        lambda t: bool(re.search(r"Nd\s*\d*.{0,4}Fe\s*\d*.{0,4}B", t) or "NdFeB" in t))

    df = df[["source_pdf", "page", "table_index", "row_index", "n_cols", "cells",
             "is_magnetic_table", "magnetic_units", "has_property_word",
             "mentions_Nd_Fe", "mentions_NdFeB"]]
    df.to_csv(PATH, index=False, encoding="utf-8-sig")

    n_tab = df.groupby(["source_pdf", "page", "table_index"]).ngroups
    n_mag = df[df.is_magnetic_table].groupby(
        ["source_pdf", "page", "table_index"]).ngroups
    print(f"tables total={n_tab}  tables with magnetic content={n_mag}")
    print(f"rows={len(df)}  magnetic rows={int(df.is_magnetic_table.sum())}  "
          f"Nd-Fe-B rows={int(df.mentions_NdFeB.sum())}")
    top = (df[df.is_magnetic_table]
           .groupby("source_pdf").size().sort_values(ascending=False).head(8))
    print(top.to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
