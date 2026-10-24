# 本文件负责读取原始数据、统一化学式、生成公共成分特征并记录清洗日志。
"""三个物理量共用的数据处理工具。"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import featlib  # noqa: E402


FEATURE_COLS = [
    "frac_Nd", "frac_Pr", "frac_Dy", "frac_Tb", "frac_Ce", "frac_Sm",
    "frac_Gd", "frac_Y", "frac_Ho", "frac_Er", "frac_La", "frac_Lu",
    "frac_Co", "frac_Cu", "frac_Ga", "frac_Al", "frac_Si", "frac_V",
    "frac_Ti", "frac_Nb", "frac_Zr", "frac_Mn", "frac_Ni", "frac_Cr",
    "frac_Fe", "frac_B", "frac_other",
    "re_frac", "tm_frac", "b_frac", "re_over_metal", "b_over_re",
    "fe_over_re", "co_over_tm", "hre_over_re", "lre_over_re",
    "w_mean_z", "w_mean_mass", "w_mean_radius", "w_mean_electroneg",
    "w_mean_valence", "mix_entropy", "size_mismatch", "electroneg_spread",
    "re_avg_degennes", "re_avg_moment", "tm_avg_moment", "has_re", "has_hre",
    "hre_mass_frac", "frac_sum_known", "untracked_frac", "is_oxide",
    "is_carbide",
]


def read_raw(relative_path: str) -> pd.DataFrame:
    """读取原始或已清洗数据，只读，不改源文件。"""
    return pd.read_csv(ROOT / relative_path, low_memory=False)


def file_hash(path: Path) -> str:
    """计算文件校验值，供结果追溯。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reduced_formula(value) -> str | None:
    """把化学式转成还原化学式。"""
    if pd.isna(value):
        return None
    try:
        return Composition(str(value)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def chemical_system(formula: str) -> str:
    """用元素集合作为分组键，避免同一体系跨训练折和验证折。"""
    try:
        return "-".join(sorted(element.symbol for element in Composition(formula).elements))
    except Exception:  # noqa: BLE001
        return "PARSE_ERROR"


def add_log(
    logs: list[dict], step: str, rows_before: int, rows_after: int, note: str
) -> None:
    """追加一条数据变化记录。"""
    logs.append(
        {
            "step": step,
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
            "removed": int(rows_before - rows_after),
            "note": note,
        }
    )


def normalize_table(
    raw: pd.DataFrame,
    formula_col: str,
    value_col: str,
    logs: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """统一化学式，并以中位数合并同一配方的重复标签。"""
    work = raw.copy()
    work["_formula"] = work[formula_col].map(reduced_formula)
    work["_value"] = pd.to_numeric(work[value_col], errors="coerce")

    before = len(work)
    work = work[work["_formula"].notna() & work["_value"].notna()]
    add_log(logs, "parse_formula_and_target", before, len(work), "删除无法解析或标签缺失的记录")

    grouped = (
        work.groupby("_formula", sort=False)
        .agg(
            y=("_value", "median"),
            y_min=("_value", "min"),
            y_max=("_value", "max"),
            n_records=("_value", "size"),
        )
        .reset_index()
        .rename(columns={"_formula": "formula"})
    )
    grouped["chem_sys"] = grouped["formula"].map(chemical_system)
    grouped["y_spread"] = grouped["y_max"] - grouped["y_min"]

    add_log(
        logs,
        "merge_duplicate_formula",
        len(work),
        len(grouped),
        "同一还原化学式保留标签中位数，原始记录数和极差保留在表中",
    )

    conflicts = grouped[grouped["y_spread"] > 1e-12].copy()
    conflicts = conflicts.sort_values("y_spread", ascending=False)
    return grouped.reset_index(drop=True), conflicts.reset_index(drop=True)


def add_features(table: pd.DataFrame) -> pd.DataFrame:
    """复用 scripts/featlib.py，保证 Demo 与其他脚本使用同一套成分特征。"""
    features = featlib.build_features(table["formula"]).reset_index(drop=True)
    missing = sorted(set(FEATURE_COLS) - set(features.columns))
    if missing:
        raise ValueError(f"featlib 缺少预期特征：{missing}")

    output = pd.concat(
        [table.reset_index(drop=True), features[FEATURE_COLS].reset_index(drop=True)],
        axis=1,
    )
    output[FEATURE_COLS] = output[FEATURE_COLS].astype(float)
    return output
