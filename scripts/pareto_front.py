"""Pareto 前沿：在"高饱和磁化 Ms / 低原料成本"之间找非支配配方。

读入
----
data/18_features_all.csv  特征矩阵（scripts/feature_engineering.py 产出）

产出
----
data/21_pareto_front.csv  非支配候选配方（按成本升序），附稳定性标记
data/21_pareto_front.png  Ms—成本散点图，前沿高亮

口径
----
    - 只对设计空间里可用的金属间化合物取前沿：ordering ∈ {FM, FiM} 且
      flagged_exotic=0（剔除氧化物/硼酸盐与成分不闭合）。这正是基线建模的同一档。
    - 两个目标：最大化 Ms_T，最小化 raw_cost_index_usd_per_kg（量级参考，非报价）。
    - 稳定性不作为目标，只作为标注：is_stable 与 energy_above_hull_eV 一并列出，
      企业落地时再叠加 hull<=0.05 eV/atom 的硬约束即可。
用法
----
python scripts/pareto_front.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def pareto_mask(ms: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """返回非支配布尔掩码：Ms 越大越好，cost 越小越好。"""
    n = len(ms)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            # j 支配 i：Ms 不更差、cost 不更高，且至少一项严格更好
            if ms[j] >= ms[i] and cost[j] <= cost[i] and (ms[j] > ms[i] or cost[j] < cost[i]):
                keep[i] = False
                break
    return keep


def main() -> int:
    feat_path = os.path.join(DATA_DIR, "18_features_all.csv")
    if not os.path.exists(feat_path):
        print(f"找不到 {feat_path}，先跑 scripts/feature_engineering.py")
        return 1
    df = pd.read_csv(feat_path, low_memory=False)
    ds = df[(df["pool"] == "design_space")
            & (df["ordering"].isin(["FM", "FiM"]))
            & (df["flagged_exotic"] == 0)].copy()
    ds = ds.dropna(subset=["Ms_T", "raw_cost_index_usd_per_kg"])
    print(f"可用金属间化合物 {len(ds)} 行（FM/FiM 且非氧化物/非成分不闭合）")

    ms = ds["Ms_T"].to_numpy(dtype=float)
    cost = ds["raw_cost_index_usd_per_kg"].to_numpy(dtype=float)
    keep = pareto_mask(ms, cost)
    front = ds.loc[keep].copy()
    front["pareto_rank"] = 1
    front = front.sort_values("raw_cost_index_usd_per_kg", kind="mergesort")

    cols = ["formula_pretty", "chemsys", "Ms_T", "raw_cost_index_usd_per_kg",
            "is_stable", "energy_above_hull_eV", "ordering", "hre_mass_frac",
            "re_over_metal", "pareto_rank"]
    out = front[[c for c in cols if c in front.columns]]
    out.to_csv(os.path.join(DATA_DIR, "21_pareto_front.csv"),
               index=False, encoding="utf-8-sig")
    print(f"WROTE 21_pareto_front.csv  {len(out)} 个非支配配方")
    print(out.round(3).to_string(index=False))

    # ---- 画图：散点 + 高亮前沿 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib 不可用，跳过绘图：{exc}")
        return 0

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(cost[~keep], ms[~keep], s=36, c="#b0b7c3", label="dominated", zorder=2)
    ax.scatter(cost[keep], ms[keep], s=60, c="#c0392b", label="Pareto front", zorder=3)
    order = np.argsort(cost[keep])
    ax.plot(cost[keep][order], ms[keep][order], color="#c0392b", lw=1.2, alpha=0.8, zorder=3)
    for _, r in front.iterrows():
        ax.annotate(r["formula_pretty"],
                    (r["raw_cost_index_usd_per_kg"], r["Ms_T"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points", alpha=0.9)
    ax.set_xlabel("raw cost index (USD/kg, order-of-magnitude)")
    ax.set_ylabel("Ms (T, DFT 0 K)")
    ax.set_title("Pareto front: maximize Ms vs minimize cost (RE-Fe-B design space)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png_path = os.path.join(DATA_DIR, "21_pareto_front.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"WROTE {png_path}")

    # ---- 登记 ----
    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keep_idx = idx[~idx["file"].isin(["21_pareto_front.csv"])]
        new = pd.DataFrame([{
            "file": "21_pareto_front.csv", "layer": "优化", "rows": len(out),
            "columns": len(out.columns), "source": "由 18 派生（scripts/pareto_front.py）",
            "purpose": "Ms—成本双目标的非支配配方集合（Pareto 前沿）",
            "refresh_note": "重跑 scripts/pareto_front.py 刷新"}])
        pd.concat([keep_idx, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print("登记 -> 00_dataset_index.csv")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
