# 本文件按成分分数在数据库中寻找与输入配方最接近的已测材料，供网页对照表使用。
"""最近配方检索。

三套数据库各自建索引：
  Ms  -> NEMAD/processed/ms_cleaned.csv（Ms_conv_status=ok）
  Tc  -> NEMAD/processed/tc_dataset.csv（tc_method=exp）
  Hcj -> NEMAD/processed/hcj_cleaned.csv（剔除 likely_anisotropy_field）

相似度用原子分数向量的欧氏距离；同一约化化学式取标签中位数。
距离为 0 表示库里有同一配方，此时实测值与预测值的差才是真正的模型误差。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Demo.common.data import reduced_formula  # noqa: E402

SOURCES = {
    "Ms": ("dataset/NEMAD/processed/ms_cleaned.csv", "material_name", "Ms_emu_g"),
    "Tc": ("dataset/NEMAD/processed/tc_dataset.csv", "formula", "tc_K"),
    "Hcj": ("dataset/NEMAD/processed/hcj_cleaned.csv", "material_name", "Hcj_kA_m"),
}


def _composition(formula: str) -> dict | None:
    """原子分数字典；解析失败返回 None。"""
    try:
        return {str(el): float(frac) for el, frac in Composition(formula).fractional_composition.as_dict().items()}
    except Exception:  # noqa: BLE001
        return None


def _grouped(prop: str) -> list[tuple[str, float, int, dict]]:
    """读取一个属性库，按约化化学式聚合，返回 (配方, 中位数, 记录数, 成分)。"""
    path, formula_col, value_col = SOURCES[prop]
    raw = pd.read_csv(ROOT / path, low_memory=False)
    if prop == "Ms":
        raw = raw[raw["Ms_conv_status"].eq("ok")]
    elif prop == "Tc":
        raw = raw[raw["tc_method"].eq("exp")]
    elif prop == "Hcj":
        raw = raw[raw["hcj_value_class"].ne("likely_anisotropy_field")]
    raw = raw[raw[value_col].notna() & (raw[value_col] > 0)].copy()
    raw["_formula"] = raw[formula_col].map(reduced_formula)
    raw = raw[raw["_formula"].notna()]

    grouped = (
        raw.groupby("_formula")
        .agg(value=(value_col, "median"), n_records=(value_col, "size"))
        .reset_index()
    )
    rows = []
    for _, row in grouped.iterrows():
        comp = _composition(row["_formula"])
        if comp:
            rows.append((row["_formula"], float(row["value"]), int(row["n_records"]), comp))
    return rows


class NeighborIndex:
    """一个属性的最近配方索引。"""

    def __init__(self, rows: list[tuple[str, float, int, dict]]):
        self.formulas = [row[0] for row in rows]
        self.values = np.array([row[1] for row in rows], dtype=float)
        self.counts = np.array([row[2] for row in rows], dtype=int)
        self._row_of = {formula: i for i, formula in enumerate(self.formulas)}
        self.elements = sorted({el for row in rows for el in row[3]})
        self._column = {el: j for j, el in enumerate(self.elements)}
        self.matrix = np.zeros((len(rows), len(self.elements)), dtype=float)
        for i, row in enumerate(rows):
            for el, frac in row[3].items():
                self.matrix[i, self._column[el]] = frac

    def query(self, formula: str) -> dict | None:
        """返回 {formula, value, distance, exact, n_records}；无可用数据时返回 None。"""
        reduced = reduced_formula(formula)
        if reduced is None:
            return None
        if reduced in self._row_of:
            i = self._row_of[reduced]
            return {
                "formula": self.formulas[i],
                "value": float(self.values[i]),
                "distance": 0.0,
                "exact": True,
                "match_mode": "exact",
                "n_records": int(self.counts[i]),
            }
        comp = _composition(reduced)
        if comp is None or not self.formulas:
            return None
        query = np.zeros(len(self.elements), dtype=float)
        missing_sq = 0.0
        for el, frac in comp.items():
            if el in self._column:
                query[self._column[el]] = frac
            else:
                missing_sq += frac * frac
        # 库中没有的元素按距离 0 的对应项处理，补上这部分平方差
        distance = np.sqrt(((self.matrix - query) ** 2).sum(axis=1) + missing_sq)

        # 优先在和输入元素集合兼容（候选包含输入的全部元素）的配方里找，
        # 避免把缺少 Ce/Tb 等关键元素的材料当成"最近"。
        needed = [self._column[el] for el in comp if el in self._column]
        present = self.matrix[:, needed] > 0 if needed else np.zeros((len(self.formulas), 0), bool)
        compatible = present.all(axis=1) if needed else np.ones(len(self.formulas), bool)
        mode = "nearest"
        if compatible.any():
            distance = np.where(compatible, distance, np.inf)
            mode = "superset" if not compatible.all() else "nearest"

        i = int(np.argmin(distance))
        return {
            "formula": self.formulas[i],
            "value": float(self.values[i]),
            "distance": float(round(distance[i], 4)),
            "exact": False,
            "match_mode": mode,
            "n_records": int(self.counts[i]),
            "unknown_fraction": float(round(1.0 - query.sum(), 4)),
        }


_INDICES: dict[str, NeighborIndex] | None = None


def get_indices() -> dict[str, NeighborIndex]:
    """惰性构建三个索引（首次调用约 0.3 s）。"""
    global _INDICES
    if _INDICES is None:
        _INDICES = {prop: NeighborIndex(_grouped(prop)) for prop in SOURCES}
    return _INDICES
