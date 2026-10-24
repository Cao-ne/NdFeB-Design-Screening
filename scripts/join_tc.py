"""把外部居里温度（Tc）数据按成分匹配到我们的设计空间，产出可直接套用的标签表。

读入
----
data/18_features_all.csv                                       我们的特征矩阵（设计空间）
../02_Devs/Basis_Of_AI_Teamwork/Tc/tc_repm_deliverables/data/tc_dataset.csv  外部 Tc 数据

产出
----
data/22_tc_join_design_space.csv  设计空间 104 行 + Tc 标签列（未匹配处留空）

口径
----
    - 用还原化学式（pymatgen Composition.reduced_formula）做成分级匹配，跨写法对齐。
    - 同一化学式的多个文献值取中位数 tc_K_median，并保留 n_sources 作为置信度；
      建模时仍要按化学式分组切分（外部数据自带的提示）。
    - 事先剔除外部数据已知的 6 条上游错误值（Nd2Fe14B 450/768.15、Nd2Fe17 743、
      Ce2Fe14B 490、Er2Fe14B 314、Y2Co17 940）。
用法
----
python scripts/join_tc.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from pymatgen.core import Composition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TC_PATH = r"C:\Users\Administrator\Desktop\02_Devs\Basis_Of_AI_Teamwork\Tc\tc_repm_deliverables\data\tc_dataset.csv"

# 外部数据 QC 已确认的上游错误值（见其 data_dictionary §6.2），套用前剔除
FLAGGED = {
    ("Nd2Fe14B", 450.0), ("Nd2Fe14B", 768.15), ("Nd2Fe17", 743.0),
    ("Ce2Fe14B", 490.0), ("Er2Fe14B", 314.0), ("Y2Co17", 940.0),
}


def reduced(formula):
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    feat_path = os.path.join(DATA_DIR, "18_features_all.csv")
    if not os.path.exists(feat_path):
        print(f"找不到 {feat_path}，先跑 scripts/feature_engineering.py")
        return 1
    if not os.path.exists(TC_PATH):
        print(f"找不到外部 Tc 数据：{TC_PATH}")
        return 1

    feat = pd.read_csv(feat_path, low_memory=False)
    tc = pd.read_csv(TC_PATH, low_memory=False)
    tc = tc[~tc.apply(lambda r: (str(r["formula"]), float(r["tc_K"])) in FLAGGED,
                      axis=1)].copy()
    tc["_red"] = tc["formula"].map(reduced)
    feat["_red"] = feat["formula_pretty"].map(reduced)

    agg = tc.dropna(subset=["_red"]).groupby("_red", as_index=False).agg(
        tc_K_median=("tc_K", "median"),
        tc_n_sources=("tc_K", "size"),
        tc_method=("tc_method", lambda s: ",".join(sorted(set(s.dropna())))),
        tc_structure_type=("structure_type",
                           lambda s: ",".join(sorted(set(s.dropna())))),
    )
    agg["tc_K_median"] = agg["tc_K_median"].round(2)

    ds = feat[feat["pool"] == "design_space"].copy()
    joined = ds.merge(agg[["_red", "tc_K_median", "tc_n_sources", "tc_method",
                           "tc_structure_type"]], on="_red", how="left")
    joined["tc_matched"] = joined["tc_K_median"].notna()
    joined = joined.sort_values("Ms_T", ascending=False).reset_index(drop=True)

    keep = ["material_id", "formula_pretty", "chemsys", "ordering", "Ms_T",
            "is_stable", "energy_above_hull_eV", "raw_cost_index_usd_per_kg",
            "tc_matched", "tc_K_median", "tc_n_sources", "tc_method",
            "tc_structure_type"]
    out = joined[[c for c in keep if c in joined.columns]]
    out.to_csv(os.path.join(DATA_DIR, "22_tc_join_design_space.csv"),
               index=False, encoding="utf-8-sig")

    n_match = int(out["tc_matched"].sum())
    print(f"WROTE 22_tc_join_design_space.csv  设计空间 {len(out)} 行，"
          f"其中 {n_match} 行匹配到 Tc")
    print("\n匹配到的行（按 Ms 降序）：")
    print(out[out["tc_matched"]].round(3).to_string(index=False))
    print("\n未匹配行数：", int((~out["tc_matched"]).sum()))

    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keep_idx = idx[~idx["file"].isin(["22_tc_join_design_space.csv"])]
        new = pd.DataFrame([{
            "file": "22_tc_join_design_space.csv", "layer": "标签",
            "rows": len(out), "columns": len(out.columns),
            "source": "18 + 外部 Tc 数据集按成分匹配（scripts/join_tc.py）",
            "purpose": "设计空间附加居里温度标签（同化学式多值取中位数，未匹配留空）",
            "refresh_note": "重跑 scripts/join_tc.py 刷新"}])
        pd.concat([keep_idx, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print("登记 -> 00_dataset_index.csv")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
