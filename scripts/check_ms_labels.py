"""核对 DFT 设计空间的 Ms 标签：与外部实验数据逐组分比对。

背景
----
data/16/18 里的 Ms_T 来自 Materials Project 的 total_magnetization。对重稀土
2:14:1 化合物，MP 的总磁矩几乎与 Nd2Fe14B 相同，而实验值只有其三分之一左右——
说明那些条目没有把重稀土 4f 与 Fe 的反平行耦合算进去。

产出
----
data/29_ms_label_conflict.csv  2:14:1 家族逐组分的 DFT vs 实验磁矩比对

用法
----
python scripts/check_ms_labels.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MS_EXT = (r"C:\Users\Administrator\Desktop\02_Devs\Basis_Of_AI_Teamwork"
          r"\Ms\outputs\Ms_cleaned.csv")
MU_B_PER_EMU_G = 5585.0  # 1 μB/f.u. = 5585 emu/mol


def reduced(formula):
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    if not os.path.exists(MS_EXT):
        print(f"找不到外部 Ms 数据：{MS_EXT}")
        return 1

    # 实验侧：外部 Ms 数据里形如 RE2Fe14B 的条目，统一折算成 μB/f.u.
    ext = pd.read_csv(MS_EXT, low_memory=False)
    ext = ext[ext["Ms_muB_fu"].notna()].copy()
    ext["_red"] = ext["material_name"].map(reduced)
    ext = ext[ext["material_name"].astype(str).str.contains("Fe14B", regex=False)]
    ext_exp = (ext.groupby("_red", as_index=False)
               .agg(exp_muB_fu_median=("Ms_muB_fu", "median"),
                    exp_muB_fu_min=("Ms_muB_fu", "min"),
                    exp_muB_fu_max=("Ms_muB_fu", "max"),
                    exp_n=("Ms_muB_fu", "size")))

    # DFT 侧
    df = pd.read_csv(os.path.join(DATA_DIR, "09_mp_magnetism.csv"), low_memory=False)
    df = df[(df["pool"] == "design_space") & df["moment_uB_per_fu"].notna()].copy()
    df["_red"] = df["formula_pretty"].map(reduced)
    df = df[df["_red"].astype(str).str.contains("Fe14B", regex=False)]
    dft = (df.groupby("_red", as_index=False)
           .agg(dft_muB_fu=("moment_uB_per_fu", "max"),
                dft_Ms_T=("Ms_T", "max"),
                dft_ordering=("ordering", "first")))

    m = dft.merge(ext_exp, on="_red", how="inner")
    m["ratio_exp_over_dft"] = m["exp_muB_fu_median"] / m["dft_muB_fu"]
    m = m.sort_values("ratio_exp_over_dft")
    m = m.rename(columns={"_red": "reduced_formula"})
    m.round(3).to_csv(os.path.join(DATA_DIR, "29_ms_label_conflict.csv"),
                      index=False, encoding="utf-8-sig")

    print(f"2:14:1 家族可比对组分 {len(m)} 个")
    print(m.round(2).to_string(index=False))
    strong = m[m["ratio_exp_over_dft"] < 0.7]
    print(f"\n实验值不足 DFT 七成的组分（疑似 4f 反平行未计入）：{len(strong)} 个")
    print("  " + ", ".join(strong["reduced_formula"].tolist()))

    # DFT 与实验的 Spearman / 偏差
    rho = m[["dft_muB_fu", "exp_muB_fu_median"]].corr(method="spearman").iloc[0, 1]
    print(f"\nDFT vs 实验 磁矩的 Spearman 相关 = {rho:.3f}")
    print(f"DFT 相对实验的系统性偏高 = "
          f"{(m['dft_muB_fu'] - m['exp_muB_fu_median']).mean():+.2f} μB/f.u.")

    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keep = idx[idx["file"] != "29_ms_label_conflict.csv"]
        new = pd.DataFrame([{
            "file": "29_ms_label_conflict.csv", "layer": "审计",
            "rows": len(m), "columns": len(m.columns),
            "source": "09 + 外部 Ms 数据比对（scripts/check_ms_labels.py）",
            "purpose": "2:14:1 家族 DFT 与实验磁矩的逐组分比对，界定 DFT 标签可用范围",
            "refresh_note": "重跑 scripts/check_ms_labels.py 刷新"}])
        pd.concat([keep, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print("登记 -> 00_dataset_index.csv")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
