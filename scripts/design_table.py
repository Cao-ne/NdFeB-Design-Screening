"""统一设计表：把 Ms / Tc / Hcj / 成本 / 稳定性并到一张表，逐行标注可信度。

读入
----
data/18_features_all.csv            设计空间基础（成分、凸包能、成本、晶系）
data/28_ms_imputed_design_space.csv 成分法 Ms 预测（emu/g 与 μ0Ms）
data/24_tc_imputed_design_space.csv Tc（实验匹配值 + 模型补全）
data/27_hcj_imputed_design_space.csv Hcj 预测 + 适用域 + 硬磁家族
data/29_ms_label_conflict.csv       DFT Ms 标签的问题清单

产出
----
data/30_design_table.csv            建模主表：104 行 × 各性能 + 可信标记

口径
----
    - Ms 以"实验训练的成分类模型"为准，DFT 值只作参照。2:14:1 家族里 Ho/Dy/Tb/Er
      四个组分的 DFT 磁矩偏高一倍以上（见 29 号表），标 ms_label_status=dft_suspect。
    - 同时给 μB/f.u.（免密度，可与 DFT 的 moment_uB_per_fu 直接比较）与 μ0Ms（T，需密度）。
    - 下游优化只应使用 design_usable=1 的行。
用法
----
python scripts/design_table.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MU_B_PER_EMU_G = 5585.0
EMU_G_TO_T = 1.2566e-3


def molar_mass(formula) -> float:
    try:
        return float(Composition(str(formula)).weight)
    except Exception:  # noqa: BLE001
        return np.nan


def main() -> int:
    feat = pd.read_csv(os.path.join(DATA_DIR, "18_features_all.csv"), low_memory=False)
    ds = feat[feat["pool"] == "design_space"].copy()
    print(f"设计空间 {len(ds)} 行")

    ms = pd.read_csv(os.path.join(DATA_DIR, "28_ms_imputed_design_space.csv"),
                     low_memory=False)
    tc = pd.read_csv(os.path.join(DATA_DIR, "24_tc_imputed_design_space.csv"),
                     low_memory=False)
    hcj = pd.read_csv(os.path.join(DATA_DIR, "27_hcj_imputed_design_space.csv"),
                      low_memory=False)
    conflict_path = os.path.join(DATA_DIR, "29_ms_label_conflict.csv")
    conflict = (pd.read_csv(conflict_path, low_memory=False)
                if os.path.exists(conflict_path) else pd.DataFrame(
                    columns=["reduced_formula", "ratio_exp_over_dft"]))
    suspect = set(conflict.loc[conflict["ratio_exp_over_dft"] < 0.7,
                               "reduced_formula"]) \
        if "ratio_exp_over_dft" in conflict.columns else set()
    print(f"DFT Ms 标签存疑的组分：{sorted(suspect) if suspect else '无'}")

    keep = ["material_id", "formula_pretty", "chemsys", "ordering",
            "crystal_system", "space_group_symbol", "is_stable",
            "energy_above_hull_eV", "raw_cost_index_usd_per_kg", "hre_mass_frac",
            "re_over_metal", "b_over_re", "fe_over_re", "co_over_tm",
            "hre_over_re", "frac_Fe", "frac_Co", "frac_B", "Ms_T",
            "density_g_cm3", "moment_uB_per_fu"]
    t = ds[[c for c in keep if c in ds.columns]].copy()
    t = t.merge(ms[["material_id", "ms_emu_g_pred", "mu0_Ms_T_pred",
                    "mu0_Ms_T_dft", "gap_vs_dft_T"]],
                on="material_id", how="left")
    t = t.merge(tc[["material_id", "tc_final", "tc_source", "tc_experimental_median",
                    "tc_n_sources"]], on="material_id", how="left")
    t = t.merge(hcj[["material_id", "hcj_kA_m_pred", "hcj_pred_dex", "ad_dist",
                     "in_domain", "hard_family", "hcj_trustworthy"]],
                on="material_id", how="left")

    # μB/f.u. 由 emu/g 折算（只需摩尔质量，不需要密度）
    t["M_g_per_mol"] = t["formula_pretty"].map(molar_mass)
    t["ms_muB_fu_pred"] = (t["ms_emu_g_pred"] * t["M_g_per_mol"]
                           / MU_B_PER_EMU_G).round(2)
    t["ms_muB_fu_dft"] = t["moment_uB_per_fu"]
    t["ms_ratio_exp_over_dft"] = (t["ms_muB_fu_pred"] / t["ms_muB_fu_dft"]).round(3)

    def label_status(f):
        try:
            red = Composition(str(f)).reduced_formula
        except Exception:  # noqa: BLE001
            return "unparsed"
        return "dft_suspect" if red in suspect else "dft_ok"

    t["ms_label_status"] = t["formula_pretty"].map(label_status)
    # 下游使用的 Ms：DFT 可信则用 DFT（同源、精度高），否则用实验模型值
    t["ms_muB_fu_used"] = np.where(t["ms_label_status"] == "dft_suspect",
                                   t["ms_muB_fu_pred"], t["ms_muB_fu_dft"])
    t["ms_source_used"] = np.where(t["ms_label_status"] == "dft_suspect",
                                   "composition_model", "dft")

    t["design_usable"] = (
        t["ordering"].isin(["FM", "FiM"])
        & (t["hard_family"] != "other")
        & t["in_domain"].fillna(False)
        & t["ms_muB_fu_used"].notna()
        & t["tc_final"].notna()
    )
    t = t.sort_values(["design_usable", "ms_muB_fu_used"],
                      ascending=[False, False]).reset_index(drop=True)
    t.to_csv(os.path.join(DATA_DIR, "30_design_table.csv"),
             index=False, encoding="utf-8-sig")

    print(f"WROTE 30_design_table.csv  {len(t)} 行")
    print(f"  DFT 标签存疑行 {int((t['ms_label_status'] == 'dft_suspect').sum())}")
    print(f"  落在已知硬磁家族 {int((t['hard_family'] != 'other').sum())}")
    print(f"  在 Hcj 适用域内 {int(t['in_domain'].fillna(False).sum())}")
    print(f"  design_usable=1 {int(t['design_usable'].sum())} 行")
    show = ["formula_pretty", "hard_family", "ms_muB_fu_used", "ms_source_used",
            "tc_final", "hcj_kA_m_pred", "energy_above_hull_eV",
            "raw_cost_index_usd_per_kg"]
    print(t[t["design_usable"]][show].round(2).to_string(index=False))

    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keep_idx = idx[idx["file"] != "30_design_table.csv"]
        new = pd.DataFrame([{
            "file": "30_design_table.csv", "layer": "建模",
            "rows": len(t), "columns": len(t.columns),
            "source": "18+24+27+28+29 合并（scripts/design_table.py）",
            "purpose": "建模主表：Ms/Tc/Hcj/成本/稳定性并排，附 DFT 标签质量与可信标记",
            "refresh_note": "重跑 scripts/design_table.py 刷新"}])
        pd.concat([keep_idx, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print("登记 -> 00_dataset_index.csv")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
