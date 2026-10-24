# 本脚本用 Ha 数据对 Hcj 做物理上界校验（观察值 + 设计空间预测值），只输出校验表，不改预测。
"""Hcj <= Ha 上界校验 v0。

物理前提：内禀矫顽力不可能超过各向异性场（Brown 悖论）。
本脚本不建模、不抬高 Hcj，只回答两件事：
  1) 已有可信 Hcj 观察值与 Ha 配对后，是否都满足 Hcj <= Ha；
  2) 设计空间的 Hcj 预测值是否越过同配方的 Ha。

输入（只读）：
  dataset/shared/processed/33_ha_dataset.csv          Ha 按配方聚合（中位数）
  dataset/NEMAD/processed/hcj_cleaned.csv             观察 Hcj（只用 plausible/borderline）
  dataset/shared/processed/27_hcj_imputed_design_space.csv  设计空间 Hcj 预测

输出：
  dataset/shared/processed/34_hcj_ha_bound_check.csv  逐条校验结果
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HA_DATASET = ROOT / "dataset/shared/processed/33_ha_dataset.csv"
HCJ_CLEANED = ROOT / "dataset/NEMAD/processed/hcj_cleaned.csv"
HCJ_PRED = ROOT / "dataset/shared/processed/27_hcj_imputed_design_space.csv"
OUT_DIR = ROOT / "dataset/shared/processed"

# 与 33 号脚本一致：Hcj 主单位 kA/m，1 T = 795.7747 kA/m
KA_PER_T = 795.7747

COLUMNS = [
    "check_type",
    "formula",
    "ha_kA_m",
    "ha_T",
    "value_kA_m",
    "value_source",
    "ME",
    "margin_kA_m",
    "status",
    "ha_n_records",
    "ha_max_over_min",
    "ha_sources",
    "temp_note",
    "note",
]

DICT_ROWS = [
    ("shared/processed/34_hcj_ha_bound_check.csv", "check_type", "校验类型", "observed / design"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "formula", "约化化学式", "-"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "ha_kA_m", "该配方 Ha 中位数", "kA/m"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "ha_T", "该配方 Ha 中位数", "T"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "value_kA_m", "被校验的 Hcj 值（观察或预测）", "kA/m"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "value_source", "Hcj 值来源", "-"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "ME", "工艺效率系数 ME=Hcj/Ha（Manufacture Efficiency）", "(0,1)"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "margin_kA_m", "Ha 减 Hcj 的余量", "kA/m"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "status", "是否满足 Hcj<=Ha", "within_bound / violates_upper_bound"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "ha_n_records", "该配方 Ha 记录数", "整数"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "ha_max_over_min", "该配方 Ha 极差比", ">=1"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "ha_sources", "该配方 Ha 的来源文件集合", "-"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "temp_note", "温度口径是否对齐", "-"),
    ("shared/processed/34_hcj_ha_bound_check.csv", "note", "备注（样本数、家族、口径）", "-"),
]


def reduced_formula(value: object) -> str | None:
    """约化化学式；解析失败返回 None。"""
    from pymatgen.core import Composition

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Composition(text).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def observed_check(ha: pd.DataFrame) -> pd.DataFrame:
    """观察 Hcj（可信档）与 Ha 按配方配对，校验 Hcj<=Ha。"""
    hcj = pd.read_csv(HCJ_CLEANED, low_memory=False)
    hcj["formula"] = hcj["material_name"].map(reduced_formula)
    # 只用作废上半段之外的可信档；likely_anisotropy_field 已整体当作 Ha，不能再用作 Hcj
    base = hcj[hcj["hcj_value_class"].ne("likely_anisotropy_field")].copy()
    base = base[base["formula"].notna()]

    agg = base.groupby("formula").agg(
        value_kA_m=("Hcj_kA_m", "median"),
        n_hcj=("Hcj_kA_m", "size"),
        has_temp=("meas_temp_K", "count"),
    )

    merged = ha.merge(agg, left_on="formula", right_index=True, how="inner")
    rows = []
    for _, r in merged.iterrows():
        temp_note = "温度未对齐" if r["n_rt"] == 0 and r["has_temp"] == 0 else "部分同温"
        if r["n_rt"] > 0 and r["has_temp"] > 0:
            temp_note = "含室温记录"
        rows.append(
            {
                "check_type": "observed",
                "formula": r["formula"],
                "ha_kA_m": r["ha_kA_m_median"],
                "ha_T": r["ha_T_median"],
                "value_kA_m": r["value_kA_m"],
                "value_source": "NEMAD plausible/borderline 中位数",
                "ME": r["value_kA_m"] / r["ha_kA_m_median"],
                "margin_kA_m": r["ha_kA_m_median"] - r["value_kA_m"],
                "status": "within_bound" if r["value_kA_m"] <= r["ha_kA_m_median"] else "violates_upper_bound",
                "ha_n_records": r["n_records"],
                "ha_max_over_min": r["max_over_min"],
                "ha_sources": r["sources"],
                "temp_note": temp_note,
                "note": f"n_Hcj={int(r['n_hcj'])}；Ha 口径={r['median_basis']}",
            }
        )
    return pd.DataFrame(rows)


def design_check(ha: pd.DataFrame) -> pd.DataFrame:
    """设计空间 Hcj 预测对同配方 Ha 校验。"""
    pred = pd.read_csv(HCJ_PRED, low_memory=False)
    pred["formula"] = pred["formula_pretty"].map(reduced_formula)
    pred = pred[pred["formula"].notna()]

    merged = pred.merge(
        ha[["formula", "ha_kA_m_median", "ha_T_median", "n_records", "max_over_min", "median_basis", "sources"]],
        on="formula",
        how="inner",
    )
    rows = []
    for _, r in merged.iterrows():
        rows.append(
            {
                "check_type": "design",
                "formula": r["formula"],
                "ha_kA_m": r["ha_kA_m_median"],
                "ha_T": r["ha_T_median"],
                "value_kA_m": r["hcj_kA_m_pred"],
                "value_source": "27_hcj_imputed_design_space 预测",
                "ME": r["hcj_kA_m_pred"] / r["ha_kA_m_median"],
                "margin_kA_m": r["ha_kA_m_median"] - r["hcj_kA_m_pred"],
                "status": "within_bound" if r["hcj_kA_m_pred"] <= r["ha_kA_m_median"] else "violates_upper_bound",
                "ha_n_records": r["n_records"],
                "ha_max_over_min": r["max_over_min"],
                "ha_sources": r["sources"],
                "temp_note": "预测按室温口径；Ha 口径见 note",
                "note": f"hard_family={r['hard_family']}；Ha 口径={r['median_basis']}；trust={r['hcj_trustworthy']}",
            }
        )
    return pd.DataFrame(rows)


def sync_index(out_path: Path) -> None:
    """把输出文件行列数补进 dataset_index.csv。"""
    index_path = ROOT / "dataset/dataset_index.csv"
    if not index_path.exists():
        return
    table = pd.read_csv(index_path)
    sample = pd.read_csv(out_path)
    name = "shared/processed/34_hcj_ha_bound_check.csv"
    table = table[table["file"].ne(name)]
    row = pd.DataFrame(
        [
            {
                "file": name,
                "layer": "外禀/各向异性",
                "rows": len(sample),
                "columns": sample.shape[1],
                "source": "33_ha_dataset + 27_hcj 预测 + hcj_cleaned",
                "purpose": "Hcj<=Ha 物理上界校验结果（只校验，不改预测）",
                "refresh_note": "python scripts/34_hcj_ha_bound_check.py",
            }
        ]
    )
    pd.concat([table, row], ignore_index=True).to_csv(index_path, index=False, encoding="utf-8-sig")


def sync_dictionary() -> None:
    """把本脚本输出字段补进 data_dictionary.csv。"""
    path = ROOT / "dataset/data_dictionary.csv"
    table = pd.read_csv(path)
    files = {row[0] for row in DICT_ROWS}
    table = table[~table["file"].isin(files)]
    add = pd.DataFrame(DICT_ROWS, columns=["file", "column", "meaning", "unit_or_range"])
    pd.concat([table, add], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hcj<=Ha 上界校验")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ha = pd.read_csv(HA_DATASET)
    observed = observed_check(ha)
    design = design_check(ha)
    result = pd.concat([observed, design], ignore_index=True)
    result = result[COLUMNS].round(
        {"ha_kA_m": 2, "ha_T": 4, "value_kA_m": 2, "ME": 4, "margin_kA_m": 2, "ha_max_over_min": 3}
    )
    out_path = out_dir / "34_hcj_ha_bound_check.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    sync_dictionary()
    sync_index(out_path)

    print(f"校验行数 {len(result)}：观察 {len(observed)} 条，设计空间 {len(design)} 条")
    pred_rows = len(pd.read_csv(HCJ_PRED))
    print(f"设计空间 {pred_rows} 行中 {len(design)} 行有 Ha 可校验（{len(design) / pred_rows:.0%}）")
    print(f"Ha 配方 {len(ha)} 个中 {len(observed)} 个与可信 Hcj 配对")
    print(f"越界 {int(result['status'].eq('violates_upper_bound').sum())} 条")
    if len(observed):
        print(
            "观察配对 ME："
            f"min={observed['ME'].min():.4f}，"
            f"中位={observed['ME'].median():.4f}，"
            f"max={observed['ME'].max():.4f}"
        )
    if len(design):
        print(
            "设计空间预测 ME："
            f"min={design['ME'].min():.4f}，"
            f"中位={design['ME'].median():.4f}，"
            f"max={design['ME'].max():.4f}"
        )
    print(f"输出：{out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
