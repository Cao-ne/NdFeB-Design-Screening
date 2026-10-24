"""Element-level cost / supply-risk table.

Indicative price levels (order of magnitude, 2024-2025 market averages) used to weight
the composition optimiser towards企业可承受的配方. These are reference numbers, not
live quotes - refresh from USGS Mineral Commodity Summaries and a spot-price feed
(e.g. Asian Metal / SMM) before publishing any cost claim.

Output: data/06_element_cost_criticality.csv
"""
import csv
import os
import sys

OUT_DIR = r"C:\Users\Administrator\Desktop\大创申报\data"

# element: (indicative metal price USD/kg, class, supply-risk note)
ELEMENTS = {
    "Fe": (0.15, "bulk commodity", "无需担忧；钢/铁为大宗"),
    "B": (3.5, "bulk commodity", "硼砂/硼铁供应充足"),
    "Cu": (9.0, "base metal", "价格随铜价波动"),
    "Al": (2.6, "base metal", "供应充足"),
    "Si": (2.0, "bulk commodity", "冶金级硅供应充足"),
    "Mn": (2.2, "bulk commodity", "供应充足"),
    "Ni": (16.0, "base metal", "印尼供应主导，价格波动"),
    "Cr": (9.0, "base metal", "供应稳定"),
    "Ti": (8.0, "base metal", "海绵钛价格波动"),
    "V": (30.0, "base metal", "供应集中，价格波动"),
    "Nb": (45.0, "critical", "巴西主导，供应链高度集中"),
    "Zr": (35.0, "critical", "核级需求带动，供应集中"),
    "Ga": (300.0, "critical", "中国出口管制，战略敏感"),
    "Co": (30.0, "critical", "刚果金主导，伦理与供应风险"),
    "Ce": (3.0, "rare earth", "轻稀土，供给过剩、价格低廉"),
    "La": (3.0, "rare earth", "轻稀土，供给过剩、价格低廉"),
    "Y": (30.0, "rare earth", "中重稀土，供应偏紧"),
    "Sm": (6.0, "rare earth", "轻稀土，钐钴磁体需求"),
    "Gd": (30.0, "rare earth", "中稀土，供应偏紧"),
    "Nd": (100.0, "rare earth", "Pr/Nd 为磁体主元素，价格中枢高"),
    "Pr": (105.0, "rare earth", "与 Nd 联动，价格中枢高"),
    "Ho": (110.0, "rare earth", "重稀土，价格高、供应紧"),
    "Er": (45.0, "rare earth", "重稀土，供应偏紧"),
    "Lu": (1500.0, "rare earth", "重稀土中最昂贵之一"),
    "Dy": (350.0, "heavy rare earth", "重稀土，晶界扩散关键元素，价格高"),
    "Tb": (900.0, "heavy rare earth", "最贵的磁体用重稀土，价格波动剧烈"),
}

ROWS = [
    {
        "element": el,
        "price_usd_per_kg": price,
        "criticality_class": cls,
        "supply_risk_note": note,
        "price_basis": "indicative 2024-2025 order of magnitude",
        "authoritative_source": "USGS Mineral Commodity Summaries; Asian Metal / SMM spot price",
    }
    for el, (price, cls, note) in ELEMENTS.items()
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "06_element_cost_criticality.csv")
    cols = ["element", "price_usd_per_kg", "criticality_class", "supply_risk_note",
            "price_basis", "authoritative_source"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(ROWS)
    print(f"WROTE {path} rows={len(ROWS)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
