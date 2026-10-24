"""成分 -> 描述符 的可复用特征模块。

供两个场景共用同一套特征空间：
    - 本项目的特征工程（scripts/feature_engineering.py 用 12 号表的原子分数）
    - 外部 Tc 数据集（scripts/train_tc.py 用化学式解析出的原子分数）
这里只负责"给定化学式 -> 原子分数 + 比值 + 加权描述符 + 磁性先验 + 质量分数 + 质量标记"，
不读写任何文件。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pymatgen.core import Composition

TRACKED = ["Nd", "Pr", "Dy", "Tb", "Ce", "Sm", "Gd", "Y", "Ho", "Er", "La", "Lu",
           "Co", "Cu", "Ga", "Al", "Si", "V", "Ti", "Nb", "Zr", "Mn", "Ni",
           "Cr", "Fe", "B"]

LREE = {"La", "Ce", "Pr", "Nd", "Sm"}
HREE = {"Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"}
TM_MAG = {"Fe": 2.22, "Co": 1.72, "Ni": 0.61}

VALENCE = {"Fe": 8, "Co": 9, "Ni": 10, "B": 3, "Nd": 3, "Pr": 3, "Dy": 3, "Tb": 3,
           "Ce": 3, "Sm": 3, "Gd": 3, "Y": 3, "Ho": 3, "Er": 3, "La": 3, "Lu": 3,
           "Cu": 11, "Ga": 3, "Al": 3, "Si": 4, "V": 5, "Ti": 4, "Nb": 5, "Zr": 4,
           "Mn": 7, "Cr": 6}

DE_GENNES = {"Ce": 0.1786, "Pr": 0.8, "Nd": 1.8409, "Sm": 4.4643, "Gd": 15.75,
             "Tb": 10.5, "Dy": 7.0833, "Ho": 4.5, "Er": 2.55, "Tm": 1.1667,
             "Yb": 0.3214, "Lu": 0.0, "Y": 0.0, "La": 0.0}
RE_MOMENT = {"Ce": 2.54, "Pr": 3.58, "Nd": 3.62, "Sm": 0.84, "Gd": 7.94,
             "Tb": 9.72, "Dy": 10.63, "Ho": 10.60, "Er": 9.58, "Tm": 7.56,
             "Yb": 4.54, "Lu": 0.0, "Y": 0.0, "La": 0.0}


def element_tables() -> dict:
    z, mass, radius, en = {}, {}, {}, {}
    for el in TRACKED:
        e = Composition(el).elements[0]
        z[el] = float(e.Z)
        mass[el] = float(e.atomic_mass)
        radius[el] = float(e.atomic_radius)
        en[el] = float(e.X)
    return {"z": z, "mass": mass, "radius": radius, "electroneg": en}


def formula_to_frac(formulas) -> pd.DataFrame:
    """化学式 -> 原子分数（TRACKED 元素 + 未跟踪元素合计）。"""
    out = {f"frac_{el}": [] for el in TRACKED}
    out["frac_other"] = []
    for f in formulas:
        comp = Composition(str(f))
        present = {e.symbol: comp.get_atomic_fraction(e.symbol) for e in comp.elements}
        for el in TRACKED:
            out[f"frac_{el}"].append(present.get(el, 0.0))
        out["frac_other"].append(1.0 - sum(present.get(el, 0.0) for el in TRACKED))
    return pd.DataFrame(out, index=formulas.index)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den.abs() > 1e-9, np.nan)


def build_features(formulas, prices: dict | None = None) -> pd.DataFrame:
    """由化学式构造完整特征表。prices 给定时额外算成本指数。"""
    f = formula_to_frac(pd.Series(formulas))
    df = pd.DataFrame(index=f.index)
    for c in f.columns:
        df[c] = f[c]

    # 化学计量比
    re_frac = f[[f"frac_{e}" for e in TRACKED if e in LREE or e in HREE]].sum(axis=1)
    hre_frac = f[[f"frac_{e}" for e in TRACKED if e in HREE]].sum(axis=1)
    lre_frac = f[[f"frac_{e}" for e in TRACKED if e in LREE]].sum(axis=1)
    tm_frac = f["frac_Fe"] + f["frac_Co"] + f["frac_Ni"]
    df["re_frac"] = re_frac
    df["tm_frac"] = tm_frac
    df["b_frac"] = f["frac_B"]
    df["re_over_metal"] = _safe_div(re_frac, re_frac + tm_frac)
    df["b_over_re"] = _safe_div(f["frac_B"], re_frac)
    df["fe_over_re"] = _safe_div(f["frac_Fe"], re_frac)
    df["co_over_tm"] = _safe_div(f["frac_Co"], tm_frac)
    df["hre_over_re"] = _safe_div(hre_frac, re_frac)
    df["lre_over_re"] = _safe_div(lre_frac, re_frac)

    # 加权描述符
    props = element_tables()
    known_sum = f[[f"frac_{e}" for e in TRACKED]].sum(axis=1).clip(lower=1e-9)
    for name, table in [("z", props["z"]), ("mass", props["mass"]),
                        ("radius", props["radius"]), ("electroneg", props["electroneg"])]:
        df[f"w_mean_{name}"] = sum(f[f"frac_{e}"] * table[e] for e in TRACKED) / known_sum
    df["w_mean_valence"] = sum(f[f"frac_{e}"] * VALENCE[e] for e in TRACKED) / known_sum
    df["mix_entropy"] = -sum(f[f"frac_{e}"] * np.log(f[f"frac_{e}"].clip(lower=1e-12))
                             for e in TRACKED) \
        - f["frac_other"] * np.log(f["frac_other"].clip(lower=1e-12))
    df["size_mismatch"] = np.sqrt(sum(
        f[f"frac_{e}"] * (props["radius"][e] - df["w_mean_radius"]) ** 2
        for e in TRACKED) / known_sum)
    df["electroneg_spread"] = np.sqrt(sum(
        f[f"frac_{e}"] * (props["electroneg"][e] - df["w_mean_electroneg"]) ** 2
        for e in TRACKED) / known_sum)

    # 磁性先验
    df["re_avg_degennes"] = sum(f[f"frac_{e}"] * DE_GENNES.get(e, 0.0)
                                for e in TRACKED) / re_frac.clip(lower=1e-9)
    df["re_avg_moment"] = sum(f[f"frac_{e}"] * RE_MOMENT.get(e, 0.0)
                              for e in TRACKED) / re_frac.clip(lower=1e-9)
    df["tm_avg_moment"] = (f["frac_Fe"] * TM_MAG["Fe"] + f["frac_Co"] * TM_MAG["Co"]
                           + f["frac_Ni"] * TM_MAG["Ni"]) / tm_frac.clip(lower=1e-9)
    df["has_re"] = (re_frac > 1e-9).astype(int)
    df["has_hre"] = (df["hre_over_re"].fillna(0.0) > 1e-9).astype(int)

    # 质量标记与成分闭合度
    df["frac_sum_known"] = known_sum.round(4)
    df["untracked_frac"] = f["frac_other"]
    symbols = [Composition(str(x)).elements for x in formulas]
    df["is_oxide"] = [any(e.symbol == "O" for e in s) for s in symbols]
    df["is_carbide"] = [any(e.symbol == "C" for e in s) for s in symbols]

    # 质量分数与成本
    mass = {e: f[f"frac_{e}"] * props["mass"][e] for e in TRACKED}
    mass_sum = sum(mass.values()).clip(lower=1e-9)
    for e in TRACKED:
        df[f"mass_frac_{e}"] = (mass[e] / mass_sum).round(6)
    df["hre_mass_frac"] = sum(df[f"mass_frac_{e}"] for e in TRACKED if e in HREE)
    if prices is not None:
        df["raw_cost_index_usd_per_kg"] = sum(
            df[f"mass_frac_{e}"] * prices.get(e, 0.0) for e in TRACKED).round(2)
    return df
