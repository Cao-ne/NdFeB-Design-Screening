"""Assemble the deliverable set: a wide literature table, a dataset index and a
column dictionary.

Outputs:
  data/08_literature_wide.csv
  data/00_dataset_index.csv
  data/00_data_dictionary.csv
"""
import csv
import os
import sys
from collections import defaultdict

import pandas as pd

OUT_DIR = r"C:\Users\Administrator\Desktop\大创申报\data"

INDEX_DESCRIPTIONS = {
    "00_dataset_index.csv": ("清单", "本文件清单", "记录每个数据文件的来源、规模与用途"),
    "00_data_dictionary.csv": ("字典", "字段说明", "各数据文件的列含义、单位与取值口径"),
    "01_dft_structures_mp.csv": (
        "内禀/结构", "Materials Project（OPTIMADE 免密钥通道）",
        "含稀土+Fe+B 的晶体结构与形成能/凸包能，用于相稳定性与结构描述符"),
    "02_nomad_entries.csv": (
        "原始记录", "NOMAD 公开数据仓库",
        "Nd/Dy/Tb/Pr/Ce/Sm/Gd/Y + Fe + B 的条目索引（含计算/实验元数据与链接）"),
    "03_literature_records.csv": (
        "内禀+外禀", "本地 36 篇文献正文抽取",
        "成分—磁性能记录，附原文证据句与置信度，已做单位换算与合理性过滤"),
    "04_literature_data_links.csv": (
        "线索", "本地文献中出现的仓库链接",
        "Zenodo/GitHub/Materials Cloud 等可下载数据集入口（本机网络受限，未下载）"),
    "05_literature_tables.csv": (
        "原始表格", "本地 36 篇文献表格抽取",
        "PDF 表格按行长格式还原，含矫顽力/磁能积等实验数据，需二次结构化"),
    "06_element_cost_criticality.csv": (
        "成本", "元素价格与供应风险（半定量）",
        "用于把成分优化目标约束到企业可承受的配方"),
    "07_external_datasets.csv": (
        "外部数据集", "Zenodo/GitHub 下载结果",
        "本机对 github.com/zenodo.org 不可达，运行后可用于登记已取回的外部数据集"),
    "08_literature_wide.csv": (
        "建模用", "由 03 透视生成的宽表",
        "一行一个组分，各磁性能列并排，便于直接用于回归/分类建模"),
    "09_mp_magnetism.csv": (
        "内禀（DFT）", "Materials Project API（带密钥）",
        "4291 条含磁矩数据：111 条 NdFeB 设计空间 + 4180 条稀土-铁/铁-硼背景池"),
    "10_design_space_ReFeB.csv": (
        "建模用", "由 09 筛出的设计空间子集",
        "RE-Fe-B 体系一行一成分，含 Ms、磁有序、凸包能、成分分数与原料成本指数"),
    "12_clean_mp_magnetism.csv": (
        "清洗后（DFT）", "由 09 去多形体得到（scripts/clean_data.py）",
        "同一化学式只保留最稳定的多形体，附多形体条数与磁矩离散度"),
    "13_clean_dft_structures.csv": (
        "清洗后（结构）", "由 01 去多形体得到（scripts/clean_data.py）",
        "结构描述符表，同一化学式只保留最稳定的一条"),
    "14_polymorph_log.csv": (
        "审计", "12/13 去多形体的剔除明细（scripts/clean_data.py）",
        "每一条被剔除的多形体都留痕：保留谁、剔除谁、能量与磁矩差多少"),
    "15_split_table.csv": (
        "建模用", "由 12 按化学体系分组切分（scripts/clean_data.py，写死）",
        "train/val/test 归属表；生成后冻结，重跑不重新洗牌，避免同体系泄漏"),
    "16_design_space_model.csv": (
        "建模用", "12 的设计空间子集 + 15 的切分 + 06 的成本指数",
        "建模主表：一行一个组分，成分分数、磁性能、成本指数与切分归属并排"),
}

# atomic masses (g/mol) for the elements we track, used for the raw-material cost index
ATOMIC_MASS = {
    "Fe": 55.845, "B": 10.81, "Nd": 144.242, "Pr": 140.908, "Dy": 162.5,
    "Tb": 158.925, "Ce": 140.116, "Sm": 150.36, "Gd": 157.25, "Y": 88.906,
    "Ho": 164.93, "Er": 167.259, "La": 138.905, "Lu": 174.967, "Co": 58.933,
    "Cu": 63.546, "Ga": 69.723, "Al": 26.982, "Si": 28.085, "V": 50.942,
    "Ti": 47.867, "Nb": 92.906, "Zr": 91.224, "Mn": 54.938, "Ni": 58.693,
    "Cr": 51.996,
}

DICTIONARY = [
    ("01_dft_structures_mp.csv", "mp_id", "Materials Project 材料编号", "-"),
    ("01_dft_structures_mp.csv", "formula_reduced", "约化化学式", "-"),
    ("01_dft_structures_mp.csv", "frac_<El>", "各元素原子分数（组成描述符）", "0-1"),
    ("01_dft_structures_mp.csv", "space_group_symbol", "空间群符号", "-"),
    ("01_dft_structures_mp.csv", "space_group_number", "空间群编号（IT 编号）", "1-230"),
    ("01_dft_structures_mp.csv", "crystal_system_from_cell",
     "由晶胞参数派生的晶系（派生量，非实验对称性指认）", "-"),
    ("01_dft_structures_mp.csv", "a_A/b_A/c_A", "晶格常数", "Å"),
    ("01_dft_structures_mp.csv", "alpha_deg/beta_deg/gamma_deg", "晶格夹角", "度"),
    ("01_dft_structures_mp.csv", "formation_energy_per_atom_eV", "每原子形成能（GGA/GGA+U）", "eV/atom"),
    ("01_dft_structures_mp.csv", "energy_above_hull_eV", "凸包以上能量（相稳定性）", "eV/atom"),
    ("02_nomad_entries.csv", "entry_id", "NOMAD 条目编号", "-"),
    ("02_nomad_entries.csv", "results.material.chemical_formula_hill", "Hill 记法化学式", "-"),
    ("02_nomad_entries.csv", "results.material.symmetry.space_group_symbol", "空间群", "-"),
    ("02_nomad_entries.csv", "results.properties.available_properties", "该条目携带的性能字段", "-"),
    ("02_nomad_entries.csv", "rare_earth", "检索所用稀土元素", "-"),
    ("03_literature_records.csv", "property", "性能类型（Tc/Ms/BHmax/Hc/Ha/Br）", "-"),
    ("03_literature_records.csv", "composition_nearest", "离数值最近的组分，即该数值最可能对应的成分", "-"),
    ("03_literature_records.csv", "value_raw/unit_raw", "原文数值与单位", "-"),
    ("03_literature_records.csv", "value_SI/unit_SI", "换算后的 SI 数值与单位", "-"),
    ("03_literature_records.csv", "confidence", "high=组分与数值距离≤80字符；medium=同句但更远", "-"),
    ("03_literature_records.csv", "process_hint", "同句出现的工艺关键词（退火/晶界扩散等）", "-"),
    ("03_literature_records.csv", "evidence", "原文证据句（人工复核用）", "-"),
    ("05_literature_tables.csv", "cells", "整行单元格以 \" | \" 连接", "-"),
    ("06_element_cost_criticality.csv", "price_usd_per_kg", "元素参考价（2024-2025 量级，非实时报价）", "USD/kg"),
    ("06_element_cost_criticality.csv", "criticality_class", "供应风险分级", "-"),
    ("08_literature_wide.csv", "Tc_K / Ms_T / Ms_emu_g / Hc_kA_m / Ha_kA_m / Ha_T / BHmax_kJ_m3",
     "按组分汇总的磁性能列，单位见列名后缀", "见列名"),
    ("09_mp_magnetism.csv", "total_magnetization_uB_cell", "DFT 单胞总磁矩", "μB/cell"),
    ("09_mp_magnetism.csv", "moment_uB_per_fu", "每化学式单位磁矩（DFT）", "μB/f.u."),
    ("09_mp_magnetism.csv", "Ms_T", "派生饱和磁化强度 = 4πM/1e4（DFT，0 K）", "T"),
    ("09_mp_magnetism.csv", "Ms_emu_g", "派生饱和比磁化强度", "emu/g"),
    ("09_mp_magnetism.csv", "ordering", "DFT 磁有序类型（FM/FiM/AFM/NM/Unknown）", "-"),
    ("09_mp_magnetism.csv", "is_magnetic", "是否磁性", "bool"),
    ("09_mp_magnetism.csv", "energy_above_hull_eV", "凸包以上能量（相稳定性）", "eV/atom"),
    ("09_mp_magnetism.csv", "pool", "design_space=目标成分空间；context=背景/预训练池", "-"),
    ("09_mp_magnetism.csv", "frac_<El>", "原子分数（组成描述符）", "0-1"),
    ("10_design_space_ReFeB.csv", "mass_frac_<El>", "质量分数（由成分折算）", "0-1"),
    ("10_design_space_ReFeB.csv", "raw_cost_index_usd_per_kg",
     "按质量分数与元素参考价估算的原料成本指数（量级参考，非报价）", "USD/kg"),
    ("12_clean_mp_magnetism.csv", "n_polymorphs", "该化学式在原始抓取结果里的记录条数", ">=1"),
    ("12_clean_mp_magnetism.csv", "polymorph_Ms_spread_T",
     "同一化学式各多形体 Ms_T 的极差（标签噪声指标）", "T"),
    ("12_clean_mp_magnetism.csv", "polymorph_hull_spread_eV", "同一化学式各多形体凸包能的极差", "eV/atom"),
    ("12_clean_mp_magnetism.csv", "select_rule", "保留该条的依据（most_stable）", "-"),
    ("14_polymorph_log.csv", "delta_Ms_T", "被剔除条目与保留条目的 Ms_T 差（kept - dropped）", "T"),
    ("15_split_table.csv", "split", "train/val/test 归属，按化学体系分组切分（写死）", "-"),
    ("15_split_table.csv", "n_in_chemsys", "同一化学体系内的记录条数（切分时的最小单位）", ">=1"),
    ("16_design_space_model.csv", "split", "该组分的 train/val/test 归属", "-"),
]


def wide_from_records(records: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "curie_temperature_Tc": [("Tc_K", "K")],
        "saturation_magnetization_Ms": [("Ms_T", "T"), ("Ms_emu_g", "emu/g")],
        "coercivity_Hc": [("Hc_kA_m", "kA/m"), ("Hc_T", "T")],
        "anisotropy_field_Ha": [("Ha_kA_m", "kA/m"), ("Ha_T", "T")],
        "energy_product_BHmax": [("BHmax_kJ_m3", "kJ/m3")],
        "remanence_Br": [("Br_T", "T"), ("Br_emu_g", "emu/g")],
    }
    agg = defaultdict(lambda: defaultdict(list))
    srcs = defaultdict(set)
    conf = defaultdict(set)
    for _, r in records.iterrows():
        key = r["composition_nearest"]
        if not isinstance(key, str) or not key:
            continue
        for col, unit in columns.get(r["property"], []):
            if r["unit_SI"] == unit:
                agg[key][col].append(r["value_SI"])
        srcs[key].add(r["source_pdf"])
        conf[key].add(r["confidence"])

    rows = []
    for comp, vals in agg.items():
        row = {"composition": comp}
        for col, _ in [(c, u) for lst in columns.values() for c, u in lst]:
            series = vals.get(col, [])
            row[col] = round(sum(series) / len(series), 4) if series else ""
            row[f"n_{col}"] = len(series)
        row["n_sources"] = len(srcs[comp])
        row["confidence"] = "high" if "high" in conf[comp] else "medium"
        row["sources"] = "; ".join(sorted(srcs[comp]))[:300]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("n_sources", ascending=False)
    return df


def main():
    rec_path = os.path.join(OUT_DIR, "03_literature_records.csv")
    records = pd.read_csv(rec_path)
    wide = wide_from_records(records)
    wide_path = os.path.join(OUT_DIR, "08_literature_wide.csv")
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    print(f"WROTE {wide_path} rows={len(wide)} cols={len(wide.columns)}")

    # --- design-space view with a raw-material cost index -------------------
    mp_path = os.path.join(OUT_DIR, "09_mp_magnetism.csv")
    cost_path = os.path.join(OUT_DIR, "06_element_cost_criticality.csv")
    if os.path.exists(mp_path) and os.path.exists(cost_path):
        mp = pd.read_csv(mp_path)
        prices = dict(zip(*pd.read_csv(cost_path)[["element", "price_usd_per_kg"]].values.T))
        ds = mp[mp["pool"] == "design_space"].copy()
        for el in ATOMIC_MASS:
            frac_col = f"frac_{el}"
            if frac_col in ds.columns:
                ds[f"mass_frac_{el}"] = ds[frac_col] * ATOMIC_MASS[el]
        mass_cols = [c for c in ds.columns if c.startswith("mass_frac_")]
        mass_sum = ds[mass_cols].sum(axis=1).replace(0, pd.NA)
        for c in mass_cols:
            ds[c] = (ds[c] / mass_sum).round(6)
        ds["raw_cost_index_usd_per_kg"] = sum(
            ds[f"mass_frac_{el}"] * prices.get(el, 0.0) for el in ATOMIC_MASS
            if f"mass_frac_{el}" in ds.columns
        ).round(2)
        keep = ["formula_pretty", "material_id", "ordering", "moment_uB_per_fu",
                "Ms_T", "Ms_emu_g", "energy_above_hull_eV", "is_stable",
                "crystal_system", "space_group_symbol", "density_g_cm3",
                "formation_energy_per_atom_eV", "raw_cost_index_usd_per_kg",
                "chemsys", "mp_url"] + sorted(mass_cols)
        ds = ds[keep].sort_values("Ms_T", ascending=False)
        ds_path = os.path.join(OUT_DIR, "10_design_space_ReFeB.csv")
        ds.to_csv(ds_path, index=False, encoding="utf-8-sig")
        print(f"WROTE {ds_path} rows={len(ds)} cols={len(ds.columns)}")

    # --- index and dictionary (written last so they see every file) ---------
    index_rows = []
    for fname in sorted(os.listdir(OUT_DIR)):
        if not fname.endswith(".csv") or fname in {"00_dataset_index.csv"}:
            continue
        fpath = os.path.join(OUT_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            rows, cols = len(df), len(df.columns)
        except Exception:  # noqa: BLE001
            rows, cols = "", ""
        kind, source, purpose = INDEX_DESCRIPTIONS.get(fname, ("-", "-", "-"))
        index_rows.append({
            "file": fname,
            "layer": kind,
            "rows": rows,
            "columns": cols,
            "source": source,
            "purpose": purpose,
            "refresh_note": "可重跑 scripts/ 下同名脚本刷新",
        })
    idx_path = os.path.join(OUT_DIR, "00_dataset_index.csv")
    with open(idx_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "layer", "rows", "columns",
                                           "source", "purpose", "refresh_note"])
        w.writeheader()
        w.writerows(index_rows)
    print(f"WROTE {idx_path} rows={len(index_rows)}")

    dic_path = os.path.join(OUT_DIR, "00_data_dictionary.csv")
    with open(dic_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "column", "meaning", "unit_or_range"])
        w.writerows(DICTIONARY)
    print(f"WROTE {dic_path} rows={len(DICTIONARY)}")

    for r in index_rows:
        print(f"  {r['file']:<34} rows={r['rows']:<6} cols={r['columns']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
