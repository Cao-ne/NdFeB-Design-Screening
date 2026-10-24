"""清洗 data/ 下抓取到的数据，产出可以直接拿去建模的表。

这个脚本只做两件事，但都是建模前必须做完的事：

1. 去多形体
   同一个化学式在 Materials Project 里经常有多条记录（不同多形体、不同计算条目、
   新旧两套编号），化学式相同但凸包能与磁矩不同。混在一起用，等于给同一个组分
   喂了互相矛盾的标签，是标签噪声最主要的来源。

   保留规则（全部可比较，排序完全确定，重跑结果一致）：
       凸包能最低 → is_stable=True 优先 → 每原子形成能最低 → material_id 字典序最小
   背景池与设计空间各自独立去重（两个池子的化学式本来也不重叠）。
   被剔除的每一条都写进 data/14_polymorph_log.csv 备查，不做静默丢弃。

2. 写死切分表
   设计空间与背景池的化学体系（chemsys）高度重合，随机切分会把同一体系的多形体
   分到 train 和 test 两头，指标虚高。这里以"化学体系"为最小单位做分组切分，
   同一个体系只会落在 train / val / test 其中之一。

   切分表 data/15_split_table.csv 生成后即冻结——再次运行本脚本默认读回这张表，
   不会重新洗牌（谁在 train 谁在 test 永远一致）；只有显式加 --rebuild-split 才会重建。
   分组贪心分配，无随机数，同样的输入必然得到同样的切分。

输出
----
data/12_clean_mp_magnetism.csv    去多形体后的 DFT 磁性能主表（design_space + context）
data/13_clean_dft_structures.csv  去多形体后的结构表（OPTIMADE 通道）
data/14_polymorph_log.csv         被剔除的多形体明细
data/15_split_table.csv           写死的切分表（material_id → train/val/test）
data/16_design_space_model.csv    清洗后 + 切分 + 成本指数的建模用设计空间表

用法
----
python scripts/clean_data.py                  # 常规：清洗 + 复用已冻结的切分表
python scripts/clean_data.py --rebuild-split  # 数据换了一批，重建切分表
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# ---------------------------------------------------------------------------
# 去多形体的目标表
# ---------------------------------------------------------------------------
DEDUPE_TARGETS = [
    {
        "source": "09_mp_magnetism.csv",
        "out": "12_clean_mp_magnetism.csv",
        "id_col": "material_id",
        "formula_col": "formula_pretty",
        "pool_col": "pool",
    },
    {
        "source": "01_dft_structures_mp.csv",
        "out": "13_clean_dft_structures.csv",
        "id_col": "mp_id",
        "formula_col": "formula_reduced",
        "pool_col": None,
    },
]

SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
GROUP_COL = "chemsys"
POOL_COL = "pool"
LABEL_COL = "ordering"
LABEL_PENALTY = 0.30  # 分组时对"磁有序构成跑偏"的惩罚权重，避免某一类全挤在一个切分里
TARGET_PENALTY = 0.20  # 分组时对"饱和磁矩量级跑偏"的惩罚权重，让各切分的 Ms_T 分布接近


def line(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1. 去多形体
# ---------------------------------------------------------------------------
def dedupe_polymorphs(df: pd.DataFrame, id_col: str, formula_col: str,
                      pool_col, source_name: str):
    """同一化学式只留最稳定的一条。返回 (保留表, 剔除明细)。"""
    keys = ([pool_col] if pool_col else []) + [formula_col]
    d = df.copy()
    d["_hull"] = pd.to_numeric(d["energy_above_hull_eV"], errors="coerce")
    d["_form"] = pd.to_numeric(d["formation_energy_per_atom_eV"], errors="coerce")
    d["_ms"] = (pd.to_numeric(d["Ms_T"], errors="coerce")
                if "Ms_T" in d.columns else pd.Series(np.nan, index=d.index))
    # False=稳定，排前面；表里没有 is_stable 列时（结构表）不参与排序
    d["_not_stable"] = (~d["is_stable"].astype(bool) if "is_stable" in d.columns
                        else False)

    d = d.sort_values(keys + ["_hull", "_not_stable", "_form", id_col],
                      kind="mergesort", na_position="last")
    d["_rank"] = d.groupby(keys, sort=False).cumcount()
    d["n_polymorphs"] = d.groupby(keys, sort=False)[id_col].transform("size")
    d["_ms_spread"] = d.groupby(keys, sort=False)["_ms"].transform(
        lambda s: float(s.max() - s.min()) if s.notna().any() else np.nan)
    d["_hull_spread"] = d.groupby(keys, sort=False)["_hull"].transform(
        lambda s: float(s.max() - s.min()) if s.notna().any() else np.nan)

    kept = d[d["_rank"] == 0].copy()
    dropped = d[d["_rank"] > 0].copy()

    # 剔除明细：一条被剔除的记录一行，保留谁、剔除谁、差多少都留痕
    k = kept[["n_polymorphs"] + keys + [id_col, "_hull", "_form", "_ms"]].rename(
        columns={id_col: "kept_id", "_hull": "kept_energy_above_hull_eV",
                 "_form": "kept_formation_energy_per_atom_eV", "_ms": "kept_Ms_T"})
    g = dropped[["n_polymorphs"] + keys + [id_col, "_hull", "_form", "_ms"]].rename(
        columns={id_col: "dropped_id", "_hull": "dropped_energy_above_hull_eV",
                 "_form": "dropped_formation_energy_per_atom_eV", "_ms": "dropped_Ms_T"})
    log = g.merge(k, on=["n_polymorphs"] + keys, how="left")
    log["delta_Ms_T"] = (log["kept_Ms_T"] - log["dropped_Ms_T"]).round(4)
    log["delta_energy_above_hull_eV"] = (
        log["dropped_energy_above_hull_eV"] - log["kept_energy_above_hull_eV"]).round(6)
    log["reason"] = "同一化学式的非最稳定多形体，已由最稳定条目代表该组分"
    log = log.rename(columns={formula_col: "formula"})
    log.insert(0, "source_file", source_name)
    log_cols = ["source_file", "formula"] + ([pool_col] if pool_col else []) + [
        "n_polymorphs", "kept_id", "kept_Ms_T", "kept_energy_above_hull_eV",
        "kept_formation_energy_per_atom_eV", "dropped_id", "dropped_Ms_T",
        "dropped_energy_above_hull_eV", "dropped_formation_energy_per_atom_eV",
        "delta_Ms_T", "delta_energy_above_hull_eV", "reason"]
    log = log[log_cols].sort_values(["formula", "dropped_id"],
                                    kind="mergesort").reset_index(drop=True)

    kept["polymorph_Ms_spread_T"] = kept["_ms_spread"].round(4)
    kept["polymorph_hull_spread_eV"] = kept["_hull_spread"].round(6)
    kept["select_rule"] = "most_stable"
    kept = kept.drop(columns=[c for c in kept.columns if c.startswith("_")])
    kept = kept.reset_index(drop=True)
    return kept, log


# ---------------------------------------------------------------------------
# 2. 写死切分表
# ---------------------------------------------------------------------------
def assign_groups(df: pd.DataFrame, ratios=SPLIT_RATIOS):
    """按化学体系分组贪心切分：大组优先，落到"还欠得最多"的那个切分里。

    只用确定性的排序和比较，没有随机数，所以同样的输入永远得到同一张表。
    """
    total = len(df)
    targets = {s: ratios[s] * total for s in SPLIT_NAMES}
    global_mix = (df[LABEL_COL].value_counts(normalize=True).to_dict()
                  if LABEL_COL in df.columns else {})
    has_target = "Ms_T" in df.columns
    ms = pd.to_numeric(df["Ms_T"], errors="coerce") if has_target else None
    g_mean = float(ms.mean()) if has_target else 0.0
    g_std = float(ms.std(ddof=0)) or 1.0

    groups = []
    for name, sub in df.groupby(GROUP_COL, sort=False):
        mix = (sub[LABEL_COL].value_counts(normalize=True).to_dict()
               if LABEL_COL in sub.columns else {})
        mean = float(pd.to_numeric(sub["Ms_T"], errors="coerce").mean()) if has_target else 0.0
        groups.append((str(name), len(sub), mix, mean * len(sub)))
    groups.sort(key=lambda t: (-t[1], t[0]))  # 大组先分，同大小按名字，保证确定性

    rows = {s: 0 for s in SPLIT_NAMES}
    sums = {s: 0.0 for s in SPLIT_NAMES}
    lab = {s: defaultdict(float) for s in SPLIT_NAMES}
    assign = {}
    for name, n, mix, gsum in groups:
        best, best_score = None, None
        for s in SPLIT_NAMES:
            fill = (targets[s] - rows[s]) / targets[s]      # 还欠多少（越大越该给）
            after = rows[s] + n
            worst = 0.0
            for lab_name, w in mix.items():
                got = (lab[s][lab_name] + w * n) / max(after, 1)
                worst = max(worst, got - global_mix.get(lab_name, 0.0))
            score = fill - LABEL_PENALTY * max(worst, 0.0)
            # 顺带把饱和磁矩的量级也摊匀，免得某个切分全是低磁矩体系
            score -= TARGET_PENALTY * abs((sums[s] + gsum) / max(after, 1) - g_mean) / g_std
            if best_score is None or score > best_score + 1e-12:
                best, best_score = s, score
        assign[name] = best
        rows[best] += n
        sums[best] += gsum
        for lab_name, w in mix.items():
            lab[best][lab_name] += w * n
    return assign


def build_split_table(clean: pd.DataFrame, pool_col: str) -> pd.DataFrame:
    parts = []
    for _, sub in clean.groupby(pool_col, sort=False):
        sub = sub.copy()
        sub["split"] = sub[GROUP_COL].map(assign_groups(sub))
        sub["n_in_chemsys"] = sub.groupby(GROUP_COL)["material_id"].transform("size")
        parts.append(sub)
    tbl = pd.concat(parts, ignore_index=True)
    tbl = tbl[["material_id", "formula_pretty", GROUP_COL, pool_col, LABEL_COL,
               "Ms_T", "split", "n_in_chemsys"]]
    tbl[pool_col] = pd.Categorical(tbl[pool_col],
                                   categories=["design_space", "context"], ordered=True)
    tbl["split"] = pd.Categorical(tbl["split"], categories=list(SPLIT_NAMES), ordered=True)
    return tbl.sort_values([pool_col, "split", GROUP_COL, "formula_pretty"],
                           kind="mergesort").reset_index(drop=True)


def verify_split(clean: pd.DataFrame, split: pd.DataFrame, pool_col: str) -> None:
    line("切分表自检")
    merged = clean.merge(split[["material_id", "split"]].astype({"split": str}),
                         on="material_id", how="left")
    if merged["split"].isna().any():
        print(f"警告：{int(merged['split'].isna().sum())} 条记录没有切分归属")
    leak = int((merged.groupby(GROUP_COL)["split"].nunique() > 1).sum())
    print(f"同一化学体系跨多个切分的泄漏：{leak} 个（必须为 0）")
    for pool in merged[pool_col].dropna().unique():
        sub = merged[merged[pool_col] == pool]
        g = sub.groupby("split")
        summary = pd.DataFrame({
            "行数": g.size(),
            "占比%": (g.size() / len(sub) * 100).round(1),
            "化学体系数": g[GROUP_COL].nunique(),
            "Ms_T中位": g["Ms_T"].median().round(3),
        })
        print(f"\n[{pool}] 共 {len(sub)} 条 / {sub[GROUP_COL].nunique()} 个化学体系")
        print(summary.to_string())
        print("  磁有序构成：" + str({str(k): int(v) for k, v in
                                  sub["ordering"].value_counts().items()}))


# ---------------------------------------------------------------------------
# 3. 登记到数据清单与字典
# ---------------------------------------------------------------------------
INDEX_HEADER = ["file", "layer", "rows", "columns", "source", "purpose", "refresh_note"]
INDEX_ROWS = [
    ("12_clean_mp_magnetism.csv", "清洗后（DFT）", "由 09 去多形体得到",
     "同一化学式只保留最稳定的多形体，附多形体条数与磁矩离散度，供建模直接用"),
    ("13_clean_dft_structures.csv", "清洗后（结构）", "由 01 去多形体得到",
     "结构描述符表，同一化学式只保留最稳定的一条"),
    ("14_polymorph_log.csv", "审计", "12/13 去多形体的剔除明细",
     "每一条被剔除的多形体都留痕：保留谁、剔除谁、能量与磁矩差多少"),
    ("15_split_table.csv", "建模用", "由 12 按化学体系分组切分生成（写死）",
     "train/val/test 归属表；生成后冻结，重跑不重新洗牌，避免同体系泄漏"),
    ("16_design_space_model.csv", "建模用", "12 的设计空间子集 + 15 的切分 + 成本指数",
     "建模主表：一行一个组分，成分分数、磁性能、成本指数与切分归属并排"),
]


def build_index_rows():
    """新产出的文件的规模从文件本身读，不写死。"""
    rows = []
    for fname, layer, source, purpose in INDEX_ROWS:
        path = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            n, c = len(df), len(df.columns)
        except Exception:  # noqa: BLE001
            n, c = "", ""
        rows.append({"file": fname, "layer": layer, "rows": n, "columns": c,
                     "source": source, "purpose": purpose,
                     "refresh_note": "重跑 scripts/clean_data.py 刷新"})
    return rows

DICT_HEADER = ["file", "column", "meaning", "unit_or_range"]
DICT_ROWS = [
    {"file": "12_clean_mp_magnetism.csv", "column": "n_polymorphs",
     "meaning": "该化学式在原始抓取结果里的记录条数", "unit_or_range": ">=1"},
    {"file": "12_clean_mp_magnetism.csv", "column": "polymorph_Ms_spread_T",
     "meaning": "同一化学式各多形体 Ms_T 的极差（标签噪声指标）", "unit_or_range": "T"},
    {"file": "12_clean_mp_magnetism.csv", "column": "polymorph_hull_spread_eV",
     "meaning": "同一化学式各多形体凸包能的极差", "unit_or_range": "eV/atom"},
    {"file": "12_clean_mp_magnetism.csv", "column": "select_rule",
     "meaning": "保留该条的依据（most_stable）", "unit_or_range": "-"},
    {"file": "15_split_table.csv", "column": "split",
     "meaning": "train/val/test 归属，按化学体系分组切分（写死）", "unit_or_range": "-"},
    {"file": "15_split_table.csv", "column": "n_in_chemsys",
     "meaning": "同一化学体系内的记录条数（切分时的最小单位）", "unit_or_range": ">=1"},
    {"file": "16_design_space_model.csv", "column": "split",
     "meaning": "该组分的 train/val/test 归属", "unit_or_range": "-"},
]


def upsert_registry(path: str, key_cols, rows, header) -> None:
    """把新产出的文件登记进清单/字典，重复登记时覆盖旧行，不重复追加。"""
    if not os.path.exists(path):
        print(f"跳过登记（找不到 {os.path.basename(path)}）")
        return
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    new = [{k: str(v).strip() if v is not None else "" for k, v in r.items()} for r in rows]
    keys = {tuple(str(d[k]) for k in key_cols) for d in new}
    keep = df[~df.apply(lambda r: tuple(str(r[k]) for k in key_cols) in keys, axis=1)]
    out = pd.concat([keep, pd.DataFrame(new)[header]], ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"登记 {len(new)} 行 -> {os.path.basename(path)}（现共 {len(out)} 行）")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="清洗 data/ 下抓取到的数据")
    ap.add_argument("--rebuild-split", action="store_true",
                    help="重建切分表（默认复用已冻结的 15_split_table.csv，不重新洗牌）")
    args = ap.parse_args()

    mp_path = os.path.join(DATA_DIR, "09_mp_magnetism.csv")
    if not os.path.exists(mp_path):
        print(f"找不到 {mp_path}")
        return 1

    # ---------------- 1. 去多形体 ----------------
    line("1. 去多形体：同一化学式只保留最稳定的一条")
    log_parts = []
    clean_main = None
    for t in DEDUPE_TARGETS:
        src = os.path.join(DATA_DIR, t["source"])
        if not os.path.exists(src):
            print(f"跳过 {t['source']}（文件不存在）")
            continue
        df = pd.read_csv(src, low_memory=False)
        kept, log = dedupe_polymorphs(df, t["id_col"], t["formula_col"],
                                      t["pool_col"], t["source"])
        log_parts.append(log)

        if t["pool_col"]:
            front = [t["pool_col"], t["id_col"], t["formula_col"], GROUP_COL]
        else:
            front = [t["id_col"], t["formula_col"]]
        kept = kept[front + [c for c in kept.columns if c not in front]]
        kept.to_csv(os.path.join(DATA_DIR, t["out"]), index=False, encoding="utf-8-sig")

        n_dup = int((kept["n_polymorphs"] > 1).sum())
        print(f"\n{t['source']}: {len(df)} 行 -> {len(kept)} 行（剔除 {len(log)} 行，"
              f"{n_dup} 个化学式有多形体）")
        if t["pool_col"]:
            for pool, sub in kept.groupby(t["pool_col"]):
                print(f"   {pool:<13} {len(sub):>5} 行 / {sub[t['formula_col']].nunique():>5} 个化学式，"
                      f"其中 {int((sub['n_polymorphs'] > 1).sum()):>4} 个有多形体")
            spread = kept.loc[kept["n_polymorphs"] > 1, "polymorph_Ms_spread_T"].dropna()
            if len(spread):
                print(f"   多形体内部 Ms_T 极差：中位 {spread.median():.3f} T、"
                      f"最大 {spread.max():.3f} T（这就是去重前的标签噪声）")
            clean_main = kept
        print(f"   WROTE {t['out']}")

    if log_parts:
        log_all = pd.concat(log_parts, ignore_index=True)
        log_all.to_csv(os.path.join(DATA_DIR, "14_polymorph_log.csv"),
                       index=False, encoding="utf-8-sig")
        print(f"\n被剔除的多形体明细 -> 14_polymorph_log.csv（{len(log_all)} 行）")
    if clean_main is None:
        print("没有可用的主表，终止")
        return 1

    # ---------------- 2. 写死切分表 ----------------
    line("2. 写死切分表：按化学体系分组，切一次就冻结")
    split_path = os.path.join(DATA_DIR, "15_split_table.csv")
    if os.path.exists(split_path) and not args.rebuild_split:
        split = pd.read_csv(split_path, encoding="utf-8-sig")
        missing = sorted(set(clean_main["material_id"]) - set(split["material_id"]))
        if missing:
            print(f"已冻结的切分表缺 {len(missing)} 条记录（数据变过了），例如 {missing[:5]}")
            print("确认无误后再用 --rebuild-split 重建。")
            return 2
        print(f"复用已冻结的切分表 15_split_table.csv（{len(split)} 行，未重新洗牌）")
    else:
        split = build_split_table(clean_main, POOL_COL)
        split.to_csv(split_path, index=False, encoding="utf-8-sig")
        print(f"{'重建' if args.rebuild_split else '首次生成'}切分表 -> 15_split_table.csv"
              f"（{len(split)} 行；目标比例 7:1.5:1.5，按化学体系分组，无随机数）")
    verify_split(clean_main, split, POOL_COL)

    # ---------------- 3. 建模用设计空间表 ----------------
    line("3. 建模用设计空间表")
    ds = clean_main[clean_main[POOL_COL] == "design_space"].merge(
        split[["material_id", "split"]].astype({"split": str}), on="material_id", how="left")
    cost_path = os.path.join(DATA_DIR, "10_design_space_ReFeB.csv")
    if os.path.exists(cost_path):
        cost = pd.read_csv(cost_path, encoding="utf-8-sig")
        cost_cols = [c for c in cost.columns
                     if c.startswith("mass_frac_") or c == "raw_cost_index_usd_per_kg"]
        ds = ds.merge(cost[["material_id"] + cost_cols], on="material_id", how="left")
    else:
        print("提示：10_design_space_ReFeB.csv 不存在，成本指数列跳过")
    keep = ["formula_pretty", "material_id", "split", "ordering", "moment_uB_per_fu",
            "Ms_T", "Ms_emu_g", "energy_above_hull_eV", "is_stable", "crystal_system",
            "space_group_symbol", "density_g_cm3", "formation_energy_per_atom_eV",
            "raw_cost_index_usd_per_kg", GROUP_COL, "n_polymorphs",
            "polymorph_Ms_spread_T", "mp_url"]
    mass_cols = sorted(c for c in ds.columns if c.startswith("mass_frac_"))
    ds = ds[[c for c in keep if c in ds.columns] + mass_cols].sort_values("Ms_T", ascending=False)
    ds.to_csv(os.path.join(DATA_DIR, "16_design_space_model.csv"),
              index=False, encoding="utf-8-sig")
    print(f"WROTE 16_design_space_model.csv  {len(ds)} 行（其中 val/test "
          f"{int(ds['split'].isin(['val', 'test']).sum())} 行，训练时不要用）")

    # ---------------- 4. 登记 ----------------
    line("4. 登记到数据清单与字典")
    upsert_registry(os.path.join(DATA_DIR, "00_dataset_index.csv"), ["file"],
                    build_index_rows(), INDEX_HEADER)
    upsert_registry(os.path.join(DATA_DIR, "00_data_dictionary.csv"), ["file", "column"],
                    DICT_ROWS, DICT_HEADER)

    line("其他重复情况（仅提示，未改动）")
    lit_path = os.path.join(DATA_DIR, "03_literature_records.csv")
    if os.path.exists(lit_path):
        lit = pd.read_csv(lit_path, low_memory=False)
        print(f"03_literature_records.csv：完全重复行 {int(lit.duplicated().sum())} 行；"
              f"同组分+同性能出现多次 {int(lit.duplicated(['composition_nearest', 'property']).sum())} 行"
              "（文献抽取层的重复，08 宽表已按均值汇总，此处不动）")
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
