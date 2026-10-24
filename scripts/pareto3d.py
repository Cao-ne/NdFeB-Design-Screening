"""三目标 Pareto 筛选：最大化 Ms 与 Tc、最小化原料成本。

读入
----
data/24_tc_imputed_design_space.csv  Tc 补全后的设计空间（实验 + 模型补全）
data/18_features_all.csv             用于取 flagged_exotic 过滤标记

产出
----
data/25_pareto3d_front.csv  三目标非支配配方
data/25_pareto3d_front.png  三维散点图（前沿高亮）

口径
----
    - 候选集 = 设计空间里 FM/FiM 且非氧化物/非成分不闭合的金属间化合物（77 行）。
    - Tc 用 24 号表的 tc_final：有实验值用实验值，否则用模型补全值。
    - 稳定性不作为目标，只标注；筛选后再叠 hull<=0.05 eV/atom 的硬约束。
用法
----
python scripts/pareto3d.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def pareto_mask(ms, tc, cost):
    n = len(ms)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            better = (ms[j] > ms[i]) or (tc[j] > tc[i]) or (cost[j] < cost[i])
            if ms[j] >= ms[i] and tc[j] >= tc[i] and cost[j] <= cost[i] and better:
                keep[i] = False
                break
    return keep


def main() -> int:
    tc_path = os.path.join(DATA_DIR, "24_tc_imputed_design_space.csv")
    feat_path = os.path.join(DATA_DIR, "18_features_all.csv")
    for p in (tc_path, feat_path):
        if not os.path.exists(p):
            print(f"找不到 {p}，先跑对应脚本")
            return 1
    tc = pd.read_csv(tc_path, low_memory=False)
    feat = pd.read_csv(feat_path, low_memory=False)[["material_id", "flagged_exotic"]]
    ds = tc.merge(feat, on="material_id", how="left")
    ds = ds[(ds["ordering"].isin(["FM", "FiM"])) & (ds["flagged_exotic"] == 0)]
    ds = ds.dropna(subset=["Ms_T", "tc_final", "raw_cost_index_usd_per_kg"]).copy()
    print(f"候选集 {len(ds)} 行（FM/FiM 且非氧化物/非成分不闭合，Tc 已补全）")

    ms = ds["Ms_T"].to_numpy(dtype=float)
    tc_v = ds["tc_final"].to_numpy(dtype=float)
    cost = ds["raw_cost_index_usd_per_kg"].to_numpy(dtype=float)
    keep = pareto_mask(ms, tc_v, cost)
    front = ds.loc[keep].copy()
    front["pareto_rank"] = 1
    front = front.sort_values("raw_cost_index_usd_per_kg", kind="mergesort")

    cols = ["formula_pretty", "chemsys", "Ms_T", "tc_final", "tc_source",
            "tc_n_sources", "raw_cost_index_usd_per_kg", "is_stable",
            "energy_above_hull_eV", "ordering", "pareto_rank"]
    out = front[[c for c in cols if c in front.columns]]
    out.to_csv(os.path.join(DATA_DIR, "25_pareto3d_front.csv"),
               index=False, encoding="utf-8-sig")
    print(f"WROTE 25_pareto3d_front.csv  {len(out)} 个非支配配方")
    print(out.round(2).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib 不可用，跳过绘图：{exc}")
        return 0

    fig = plt.figure(figsize=(9, 7), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(cost[~keep], ms[~keep], tc_v[~keep], s=28, c="#b0b7c3",
               label="dominated", depthshade=False)
    ax.scatter(cost[keep], ms[keep], tc_v[keep], s=55, c="#c0392b",
               label="Pareto front", depthshade=False)
    ax.set_xlabel("cost index (USD/kg)")
    ax.set_ylabel("Ms (T)")
    ax.set_zlabel("Tc (K)")
    ax.set_title("3-objective Pareto: max Ms & Tc, min cost")
    ax.legend(loc="best")
    fig.tight_layout()
    png_path = os.path.join(DATA_DIR, "25_pareto3d_front.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"WROTE {png_path}")

    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keep_idx = idx[~idx["file"].isin(["25_pareto3d_front.csv"])]
        new = pd.DataFrame([{
            "file": "25_pareto3d_front.csv", "layer": "优化", "rows": len(out),
            "columns": len(out.columns), "source": "由 24 派生（scripts/pareto3d.py）",
            "purpose": "Ms/Tc/成本三目标的非支配配方集合",
            "refresh_note": "重跑 scripts/pareto3d.py 刷新"}])
        pd.concat([keep_idx, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print("登记 -> 00_dataset_index.csv")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
