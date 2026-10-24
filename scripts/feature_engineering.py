"""特征工程：把成分表映射成可建模的描述符，并给设计空间留一份数据质量审计。

读入
----
data/12_clean_mp_magnetism.csv   清洗后的 DFT 磁性能主表（design_space + context）
data/15_split_table.csv          冻结的 train/val/test 归属
data/06_element_cost_criticality.csv  元素价格（用于原料成本指数）

产出
----
data/17_feature_audit.csv        设计空间逐行数据质量留痕（氧化物/成分不闭合/磁有序）
data/18_features_all.csv         全量特征矩阵：3395 行，成分 + 比值 + 加权描述符 + 磁性 + 成本

特征口径（都在脚本内确定，重跑结果一致）：
    - 原子分数 frac_<El> 直接来自 12；frac_other 是未被跟踪的元素之和（O/C/Bi/Th/Sc/Tm 等）。
    - 质量分数由原子分数 x 原子量重新折算（只对已跟踪元素归一）。
    - 加权描述符（原子序数/原子量/半径/电负性/价电子）只对"已跟踪元素"做成分加权平均，
      未跟踪元素单独用 frac_other 告诉模型，避免给它编造一个物理属性。
    - 稀土分类：轻稀土 = {La, Ce, Pr, Nd, Sm}；重稀土 = {Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Y}。
    - 磁性先验只喂"每稀土平均 de Gennes 因子/平均自由离子磁矩"和"每过渡金属平均磁矩"，
      不喂总量磁矩（那是 Ms_T 的近亲，进了就是泄漏）。
用法
----
python scripts/feature_engineering.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Element
except Exception:  # noqa: BLE001
    Element = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# 12 号表里逐元素跟踪的元素（不含 frac_other）
TRACKED = ["Nd", "Pr", "Dy", "Tb", "Ce", "Sm", "Gd", "Y", "Ho", "Er", "La", "Lu",
           "Co", "Cu", "Ga", "Al", "Si", "V", "Ti", "Nb", "Zr", "Mn", "Ni",
           "Cr", "Fe", "B"]

LREE = {"La", "Ce", "Pr", "Nd", "Sm"}
HREE = {"Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"}
TM_MAG = {"Fe": 2.22, "Co": 1.72, "Ni": 0.61}

# 常用化合价近似，用于价电子浓度（VEC）一类描述符
VALENCE = {"Fe": 8, "Co": 9, "Ni": 10, "B": 3, "Nd": 3, "Pr": 3, "Dy": 3, "Tb": 3,
           "Ce": 3, "Sm": 3, "Gd": 3, "Y": 3, "Ho": 3, "Er": 3, "La": 3, "Lu": 3,
           "Cu": 11, "Ga": 3, "Al": 3, "Si": 4, "V": 5, "Ti": 4, "Nb": 5, "Zr": 4,
           "Mn": 7, "Cr": 6}

# 稀土 4f 磁先验：de Gennes 因子 (g_J-1)^2 * J(J+1)，与自由离子磁矩 g_J*sqrt(J(J+1))
DE_GENNES = {"Ce": 0.1786, "Pr": 0.8, "Nd": 1.8409, "Sm": 4.4643, "Gd": 15.75,
             "Tb": 10.5, "Dy": 7.0833, "Ho": 4.5, "Er": 2.55, "Tm": 1.1667,
             "Yb": 0.3214, "Lu": 0.0, "Y": 0.0, "La": 0.0}
RE_MOMENT = {"Ce": 2.54, "Pr": 3.58, "Nd": 3.62, "Sm": 0.84, "Gd": 7.94,
             "Tb": 9.72, "Dy": 10.63, "Ho": 10.60, "Er": 9.58, "Tm": 7.56,
             "Yb": 4.54, "Lu": 0.0, "Y": 0.0, "La": 0.0}

# 供基线脚本引用的目标列与"不得当特征"的列
TARGET_COLS = ["Ms_T", "Ms_emu_g", "moment_uB_per_fu", "is_stable",
               "energy_above_hull_eV", "formation_energy_per_atom_eV"]
LEAK_COLS = ["Ms_emu_g", "moment_uB_per_fu"]


def elem_property(el: str, attr: str, fallback: float) -> float:
    """优先 pymatgen 的周期表，缺失时退回内置近似。"""
    if Element is not None:
        try:
            value = getattr(Element(el), attr)
            return float(value)
        except Exception:  # noqa: BLE001
            pass
    return fallback


def build_property_tables() -> dict:
    """每个跟踪元素的原子序数/原子量/半径/电负性/价电子，做成 dict 供向量化。"""
    if Element is not None:
        z = {el: float(Element(el).Z) for el in TRACKED}
        mass = {el: float(Element(el).atomic_mass) for el in TRACKED}
        radius = {el: float(Element(el).atomic_radius) for el in TRACKED}
        en = {el: float(Element(el).X) for el in TRACKED}
    else:
        z = {el: 0.0 for el in TRACKED}
        mass = {el: 0.0 for el in TRACKED}
        radius = {el: 0.0 for el in TRACKED}
        en = {el: 0.0 for el in TRACKED}
    return {"z": z, "mass": mass, "radius": radius, "electroneg": en}


def safe_div(num: pd.Series, den: pd.Series, fill: float = np.nan) -> pd.Series:
    return num / den.where(den.abs() > 1e-9, np.nan)


def add_ratio_features(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """化学计量比：对应 README 的'调整元素占比/组分'两条目标。"""
    re_frac = f[[f"frac_{e}" for e in TRACKED if e in LREE or e in HREE]].sum(axis=1)
    hre_frac = f[[f"frac_{e}" for e in TRACKED if e in HREE]].sum(axis=1)
    lre_frac = f[[f"frac_{e}" for e in TRACKED if e in LREE]].sum(axis=1)
    tm_frac = (f["frac_Fe"] + f.get("frac_Co", 0.0) + f.get("frac_Ni", 0.0))
    co_frac = f.get("frac_Co", 0.0)

    df["re_frac"] = re_frac
    df["tm_frac"] = tm_frac
    df["b_frac"] = f["frac_B"]
    df["re_over_metal"] = safe_div(re_frac, re_frac + tm_frac)
    df["b_over_re"] = safe_div(f["frac_B"], re_frac)
    df["fe_over_re"] = safe_div(f["frac_Fe"], re_frac)
    df["co_over_tm"] = safe_div(co_frac, tm_frac)
    df["hre_over_re"] = safe_div(hre_frac, re_frac)
    df["lre_over_re"] = safe_div(lre_frac, re_frac)
    return df


def add_weighted_descriptors(df: pd.DataFrame, f: pd.DataFrame, props: dict) -> pd.DataFrame:
    """成分加权的物理量均值 + 混合熵 + 原子尺寸错配 + 电负性差。"""
    known_sum = f[[f"frac_{e}" for e in TRACKED]].sum(axis=1).clip(lower=1e-9)
    for name, table in [("z", props["z"]), ("mass", props["mass"]),
                        ("radius", props["radius"]), ("electroneg", props["electroneg"])]:
        df[f"w_mean_{name}"] = sum(
            f[f"frac_{e}"] * table[e] for e in TRACKED
        ) / known_sum
    df["w_mean_valence"] = sum(
        f[f"frac_{e}"] * VALENCE[e] for e in TRACKED
    ) / known_sum

    # 混合熵：已知元素 + 未跟踪元素（当成一个不可分的伪物种）
    ent = -sum(f[f"frac_{e}"] * np.log(f[f"frac_{e}"].clip(lower=1e-12))
               for e in TRACKED)
    other = f["frac_other"]
    ent = ent - other * np.log(other.clip(lower=1e-12))
    df["mix_entropy"] = ent

    # 尺寸错配 δ 与电负性差 Δχ：只对已知元素算（未跟踪元素无半径/电负性）
    df["size_mismatch"] = np.sqrt(sum(
        f[f"frac_{e}"] * (props["radius"][e] - df["w_mean_radius"]) ** 2
        for e in TRACKED
    ) / known_sum)
    df["electroneg_spread"] = np.sqrt(sum(
        f[f"frac_{e}"] * (props["electroneg"][e] - df["w_mean_electroneg"]) ** 2
        for e in TRACKED
    ) / known_sum)
    return df


def add_magnetic_features(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """磁性先验：每稀土平均 de Gennes/自由离子磁矩，每过渡金属平均磁矩。"""
    re_frac = f[[f"frac_{e}" for e in TRACKED if e in LREE or e in HREE]].sum(axis=1)
    tm_frac = (f["frac_Fe"] + f.get("frac_Co", 0.0) + f.get("frac_Ni", 0.0))

    df["re_avg_degennes"] = sum(
        f[f"frac_{e}"] * DE_GENNES.get(e, 0.0) for e in TRACKED
    ) / re_frac.clip(lower=1e-9)
    df["re_avg_moment"] = sum(
        f[f"frac_{e}"] * RE_MOMENT.get(e, 0.0) for e in TRACKED
    ) / re_frac.clip(lower=1e-9)
    df["tm_avg_moment"] = (
        f["frac_Fe"] * TM_MAG["Fe"]
        + f.get("frac_Co", 0.0) * TM_MAG["Co"]
        + f.get("frac_Ni", 0.0) * TM_MAG["Ni"]
    ) / tm_frac.clip(lower=1e-9)
    df["has_re"] = (re_frac > 1e-9).astype(int)
    df["has_hre"] = (df["hre_over_re"].fillna(0.0) > 1e-9).astype(int)
    return df


def add_cost_features(df: pd.DataFrame, f: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """质量分数 + 原料成本指数（仅对已跟踪元素）。"""
    mass = {}
    for e in TRACKED:
        mass[e] = f[f"frac_{e}"] * props["mass"][e]
    mass_sum = sum(mass.values()).clip(lower=1e-9)
    for e in TRACKED:
        df[f"mass_frac_{e}"] = (mass[e] / mass_sum).round(6)
    df["hre_mass_frac"] = sum(
        df[f"mass_frac_{e}"] for e in TRACKED if e in HREE
    )
    df["raw_cost_index_usd_per_kg"] = sum(
        df[f"mass_frac_{e}"] * prices.get(e, 0.0) for e in TRACKED
    ).round(2)
    return df


def add_quality_flags(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """数据质量标记：成分闭合度、氧化物/硼酸盐、可用的磁有序。"""
    known_sum = f[[f"frac_{e}" for e in TRACKED]].sum(axis=1)
    df["frac_sum_known"] = known_sum.round(4)
    df["untracked_frac"] = f["frac_other"]
    # chemsys 是连字符分隔的元素符号，必须整词匹配，避免把 Co 误判成含 O
    tokens = df["chemsys"].fillna("").str.split("-")
    df["is_oxide"] = tokens.map(lambda ts: "O" in ts).astype(int)
    df["is_carbide"] = tokens.map(lambda ts: "C" in ts).astype(int)
    df["flagged_nonclosure"] = (known_sum < 0.8).astype(int)
    df["flagged_exotic"] = (
        (df["is_oxide"] == 1) | (df["flagged_nonclosure"] == 1)
    ).astype(int)
    df["ordering_usable"] = df["ordering"].isin(["FM", "FiM"]).astype(int)
    return df


def line(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    mp_path = os.path.join(DATA_DIR, "12_clean_mp_magnetism.csv")
    split_path = os.path.join(DATA_DIR, "15_split_table.csv")
    cost_path = os.path.join(DATA_DIR, "06_element_cost_criticality.csv")
    for p in (mp_path, split_path, cost_path):
        if not os.path.exists(p):
            print(f"找不到 {p}")
            return 1

    global props
    props = build_property_tables()
    mp = pd.read_csv(mp_path, low_memory=False)
    split = pd.read_csv(split_path, low_memory=False)
    prices = dict(zip(*pd.read_csv(cost_path)[["element", "price_usd_per_kg"]].values.T))

    frac_cols = [f"frac_{e}" for e in TRACKED] + ["frac_other"]
    f = mp[["material_id"] + [c for c in frac_cols if c in mp.columns]].copy()
    for c in frac_cols:
        if c not in f.columns:
            f[c] = 0.0
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)

    base = mp[["material_id", "formula_pretty", "chemsys", "pool", "ordering",
               "crystal_system", "space_group_symbol", "density_g_cm3",
               "n_elements", "n_sites", "n_polymorphs", "polymorph_Ms_spread_T",
               "Ms_T", "Ms_emu_g", "moment_uB_per_fu", "is_stable",
               "energy_above_hull_eV", "formation_energy_per_atom_eV"]].copy()
    base = base.merge(split[["material_id", "split"]].astype({"split": str}),
                      on="material_id", how="left")
    for col in ("Ms_T", "Ms_emu_g", "moment_uB_per_fu", "energy_above_hull_eV",
                "formation_energy_per_atom_eV", "density_g_cm3", "n_elements",
                "n_sites", "polymorph_Ms_spread_T"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    out = base.copy()
    for e in TRACKED:
        out[f"frac_{e}"] = f[f"frac_{e}"]
    out["frac_other"] = f["frac_other"]

    line("1. 化学计量比与物理描述符")
    out = add_ratio_features(out, out)
    out = add_weighted_descriptors(out, out, props)
    out = add_magnetic_features(out, out)
    out = add_cost_features(out, out, prices)
    out = add_quality_flags(out, out)

    identity = ["material_id", "formula_pretty", "chemsys", "pool", "split",
                "ordering", "crystal_system", "space_group_symbol"]
    targets = ["Ms_T", "Ms_emu_g", "moment_uB_per_fu", "is_stable",
               "energy_above_hull_eV", "formation_energy_per_atom_eV"]
    quality = ["frac_sum_known", "untracked_frac", "is_oxide", "is_carbide",
               "flagged_nonclosure", "flagged_exotic", "ordering_usable",
               "n_elements", "n_sites", "n_polymorphs", "polymorph_Ms_spread_T",
               "density_g_cm3"]
    frac_cols_sorted = [f"frac_{e}" for e in TRACKED] + ["frac_other"]
    mass_cols_sorted = [f"mass_frac_{e}" for e in TRACKED]
    ratio_cols = ["re_frac", "tm_frac", "b_frac", "re_over_metal", "b_over_re",
                  "fe_over_re", "co_over_tm", "hre_over_re", "lre_over_re"]
    weighted_cols = ["w_mean_z", "w_mean_mass", "w_mean_radius",
                     "w_mean_electroneg", "w_mean_valence", "mix_entropy",
                     "size_mismatch", "electroneg_spread"]
    magnetic_cols = ["re_avg_degennes", "re_avg_moment", "tm_avg_moment",
                     "has_re", "has_hre"]
    cost_cols = ["hre_mass_frac", "raw_cost_index_usd_per_kg"]
    order = (identity + targets + quality + frac_cols_sorted + mass_cols_sorted
             + ratio_cols + weighted_cols + magnetic_cols + cost_cols)
    out = out[[c for c in order if c in out.columns]].copy()
    out = out.sort_values(["pool", "split", "chemsys", "formula_pretty"],
                          kind="mergesort").reset_index(drop=True)
    out.to_csv(os.path.join(DATA_DIR, "18_features_all.csv"),
               index=False, encoding="utf-8-sig")
    print(f"WROTE 18_features_all.csv  {len(out)} 行 x {len(out.columns)} 列")
    print("  池构成：" + str({k: int(v) for k, v in out["pool"].value_counts().items()}))
    print("  切分构成：" + str({k: int(v) for k, v in out["split"].value_counts().items()}))

    line("2. 设计空间数据质量审计")
    ds = out[out["pool"] == "design_space"].copy()
    ds["action"] = np.where(
        ds["is_oxide"] == 1, "exclude_oxide",
        np.where(ds["flagged_nonclosure"] == 1, "exclude_nonclosure", "keep"))
    ds = ds.sort_values(["action", "Ms_T"], ascending=[True, False],
                        kind="mergesort").reset_index(drop=True)
    audit_cols = ["material_id", "formula_pretty", "chemsys", "ordering", "Ms_T",
                  "is_stable", "frac_sum_known", "untracked_frac", "is_oxide",
                  "is_carbide", "flagged_nonclosure", "action"]
    ds[audit_cols].to_csv(os.path.join(DATA_DIR, "17_feature_audit.csv"),
                          index=False, encoding="utf-8-sig")
    print(f"WROTE 17_feature_audit.csv  {len(ds)} 行")
    print("  处理建议：" + str({k: int(v) for k, v in ds["action"].value_counts().items()}))
    print("  磁有序构成：" + str({str(k): int(v) for k, v in
                              ds["ordering"].value_counts().items()}))

    line("3. 登记到数据清单与字典")
    index_rows = [
        {"file": "17_feature_audit.csv", "layer": "审计", "rows": len(ds),
         "columns": len(audit_cols), "source": "由 12/15 派生（scripts/feature_engineering.py）",
         "purpose": "设计空间逐行数据质量留痕：氧化物/成分不闭合/磁有序，附处理建议",
         "refresh_note": "重跑 scripts/feature_engineering.py 刷新"},
        {"file": "18_features_all.csv", "layer": "特征", "rows": len(out),
         "columns": len(out.columns), "source": "由 12/15/06 派生（scripts/feature_engineering.py）",
         "purpose": "全量特征矩阵：成分分数、化学计量比、加权描述符、磁性先验与成本指数",
         "refresh_note": "重跑 scripts/feature_engineering.py 刷新"},
    ]
    dict_rows = [
        {"file": "17_feature_audit.csv", "column": "frac_sum_known",
         "meaning": "已跟踪元素原子分数之和", "unit_or_range": "0-1"},
        {"file": "17_feature_audit.csv", "column": "untracked_frac",
         "meaning": "未跟踪元素（O/C/Bi/Th/Sc/Tm 等）原子分数之和", "unit_or_range": "0-1"},
        {"file": "17_feature_audit.csv", "column": "is_oxide",
         "meaning": "化学体系含 O，多为稀土铁硼酸盐而非 RE-Fe-B 金属间化合物", "unit_or_range": "0/1"},
        {"file": "17_feature_audit.csv", "column": "action",
         "meaning": "处理建议：keep / exclude_oxide / exclude_nonclosure", "unit_or_range": "-"},
        {"file": "18_features_all.csv", "column": "re_over_metal",
         "meaning": "稀土占（稀土+Fe+Co+Ni）的原子比，区分 2:14:1 / 2:17 / 1:12", "unit_or_range": "0-1"},
        {"file": "18_features_all.csv", "column": "hre_over_re",
         "meaning": "重稀土占全部稀土的比例，矫顽力与成本的核心杠杆", "unit_or_range": "0-1"},
        {"file": "18_features_all.csv", "column": "co_over_tm",
         "meaning": "Co 在过渡金属子晶格中的取代度", "unit_or_range": "0-1"},
        {"file": "18_features_all.csv", "column": "w_mean_z/w_mean_mass/w_mean_radius/w_mean_electroneg/w_mean_valence",
         "meaning": "成分加权的平均原子序数/原子量/半径/电负性/价电子（仅已跟踪元素）", "unit_or_range": "-"},
        {"file": "18_features_all.csv", "column": "mix_entropy",
         "meaning": "成分混合熵 -Σx ln x（未跟踪元素作为单一伪物种）", "unit_or_range": "-"},
        {"file": "18_features_all.csv", "column": "size_mismatch/electroneg_spread",
         "meaning": "原子尺寸错配 δ 与电负性差 Δχ（仅已跟踪元素）", "unit_or_range": "-"},
        {"file": "18_features_all.csv", "column": "re_avg_degennes/re_avg_moment",
         "meaning": "每稀土平均 de Gennes 因子 / 自由离子磁矩（4f 磁性先验）", "unit_or_range": "-"},
        {"file": "18_features_all.csv", "column": "tm_avg_moment",
         "meaning": "每过渡金属平均原子磁矩（Fe 2.22 / Co 1.72 / Ni 0.61 加权）", "unit_or_range": "muB"},
        {"file": "18_features_all.csv", "column": "raw_cost_index_usd_per_kg",
         "meaning": "按质量分数与元素参考价估算的原料成本指数（量级参考）", "unit_or_range": "USD/kg"},
    ]
    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    dic_path = os.path.join(DATA_DIR, "00_data_dictionary.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keys = {r["file"] for r in index_rows}
        keep = idx[~idx["file"].isin(keys)]
        out_idx = pd.concat([keep, pd.DataFrame(index_rows)], ignore_index=True)
        out_idx.to_csv(idx_path, index=False, encoding="utf-8-sig")
        print(f"登记 {len(index_rows)} 行 -> 00_dataset_index.csv")
    if os.path.exists(dic_path):
        dic = pd.read_csv(dic_path, encoding="utf-8-sig", dtype=str).fillna("")
        keys = {(r["file"], r["column"]) for r in dict_rows}
        keep = dic[~dic.apply(lambda r: (str(r["file"]), str(r["column"])) in keys, axis=1)]
        out_dic = pd.concat([keep, pd.DataFrame(dict_rows)], ignore_index=True)
        out_dic.to_csv(dic_path, index=False, encoding="utf-8-sig")
        print(f"登记 {len(dict_rows)} 行 -> 00_data_dictionary.csv")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
