"""成分搜索：在 RE-Fe-B 成分空间里枚举候选配方，用三个模型预测并做多目标筛选。

这一步直接对应 README 的目标 1/2（调整元素占比、调整组分以提高永磁性能）
与目标 3（成本与关键元素约束）。

读入
----
外部 Ms / Tc / Hcj 数据（三个成分模型的训练集）
data/06_element_cost_criticality.csv  元素价格，用于算原料成本指数
data/30_design_table.csv              已确认可用的设计空间行（作适用域参照）

产出
----
data/31_search_candidates.csv    全部候选配方的预测与约束判定
data/31_search_pareto.csv        Ms / Tc / 成本三目标的非支配配方
data/31_search_pareto.png        前沿可视化
data/31_model_cache.joblib       三个模型的缓存（下次重跑直接复用）

口径
----
    - 三个模型都是纯成分模型（scripts/featlib.py 特征），只吃化学式，可对任意新成分预测。
    - 候选空间以 2:14:1 为主：RE2(Fe1-yCoy)14B，RE 由 Nd/Pr/Ce/La/Dy/Tb 配比组成。
    - Hcj 模型对 Dy/Tb 取代的排序不可信（训练集里没有重稀土 2:14:1 块体实验值，
      见 scripts/train_hcj.py 的"已知局限"），因此 Hcj 只作为展示列，不参与前沿判定。
    - 前沿只对 Ms、Tc、成本三个目标求非支配；成本越低越好，其余越高越好。
    - 适用域：候选点到训练集最近邻的标准化成分距离需落在阈值内。
用法
----
python scripts/search_compositions.py
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import featlib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
BASE_EXT = r"C:\Users\Administrator\Desktop\02_Devs\Basis_Of_AI_Teamwork"

MS_EXT = os.path.join(BASE_EXT, r"Ms\outputs\Ms_cleaned.csv")
HCJ_EXT = os.path.join(BASE_EXT, r"Hcj\outputs\Hcj_cleaned.csv")
TC_EXT = os.path.join(BASE_EXT, r"Tc\tc_repm_deliverables\data\tc_dataset.csv")

CACHE = os.path.join(DATA_DIR, "31_model_cache.joblib")
MU_B_PER_EMU_G = 5585.0

FRAC_COLS = [f"frac_{e}" for e in featlib.TRACKED] + ["frac_other"]
RATIO_COLS = ["re_frac", "tm_frac", "b_frac", "re_over_metal", "b_over_re",
              "fe_over_re", "co_over_tm", "hre_over_re", "lre_over_re"]
WEIGHTED_COLS = ["w_mean_z", "w_mean_mass", "w_mean_radius", "w_mean_electroneg",
                 "w_mean_valence", "mix_entropy", "size_mismatch",
                 "electroneg_spread"]
MAGNETIC_COLS = ["re_avg_degennes", "re_avg_moment", "tm_avg_moment",
                 "has_re", "has_hre"]
EXTRA_COLS = ["is_oxide", "is_carbide"]
FEATURES = FRAC_COLS + RATIO_COLS + WEIGHTED_COLS + MAGNETIC_COLS + EXTRA_COLS

TC_FLAGGED = {("Nd2Fe14B", 450.0), ("Nd2Fe14B", 768.15), ("Nd2Fe17", 743.0),
              ("Ce2Fe14B", 490.0), ("Er2Fe14B", 314.0), ("Y2Co17", 940.0)}


def chem_sys(f) -> str:
    try:
        return "-".join(sorted(e.symbol for e in Composition(str(f)).elements))
    except Exception:  # noqa: BLE001
        return "UNPARSED"


def build_xy(formulas, values):
    X = featlib.build_features(pd.Series(list(formulas)))[FEATURES].astype(float)
    keep = X[RATIO_COLS].notna().all(axis=1).to_numpy()
    return X[keep], np.asarray(values, dtype=float)[keep]


def train_models():
    """三个成分模型 + 适用域参照。结果缓存到 joblib。"""
    import joblib

    stamps = [os.path.getmtime(p) for p in (MS_EXT, HCJ_EXT, TC_EXT)]
    if os.path.exists(CACHE) and os.path.getmtime(CACHE) > max(stamps):
        print("加载模型缓存 31_model_cache.joblib")
        return joblib.load(CACHE)

    print("训练三个成分模型（首次运行需要几分钟）...")
    models = {}

    # --- Ms ---
    ms = pd.read_csv(MS_EXT, low_memory=False)
    ms = ms[(ms["Ms_conv_status"] == "ok") & ms["Ms_emu_g"].notna()]
    agg = (ms.groupby("material_name", as_index=False)
           .agg(v=("Ms_emu_g", "median")))
    X, y = build_xy(agg["material_name"], agg["v"])
    models["ms"] = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                             max_depth=4, min_samples_leaf=2,
                                             random_state=0).fit(X, y)
    print(f"  Ms  模型 n={len(y)}")

    # --- Tc ---
    tc = pd.read_csv(TC_EXT, low_memory=False)
    tc = tc[~tc.apply(lambda r: (str(r["formula"]), float(r["tc_K"])) in TC_FLAGGED,
                      axis=1)]
    tc = tc[tc["tc_K"] >= 10]
    agg = tc.groupby("formula", as_index=False).agg(v=("tc_K", "median"))
    X, y = build_xy(agg["formula"], agg["v"])
    models["tc"] = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                             max_depth=4, min_samples_leaf=2,
                                             random_state=0).fit(X, y)
    print(f"  Tc  模型 n={len(y)}")

    # --- Hcj ---
    hc = pd.read_csv(HCJ_EXT, low_memory=False)
    hc = hc[hc["hcj_value_class"] == "plausible_hcj"]
    agg = hc.groupby("material_name", as_index=False).agg(v=("Hcj_kA_m", "median"))
    agg = agg[agg["v"].notna()]
    X, y = build_xy(agg["material_name"], np.log10(1.0 + agg["v"].clip(lower=0)))
    models["hcj"] = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                              max_depth=4, min_samples_leaf=2,
                                              random_state=0).fit(X, y)
    print(f"  Hcj 模型 n={len(y)}")

    # --- 适用域参照：三个训练集的成分分布 ---
    ref_frames = []
    for path, col in ((MS_EXT, "material_name"), (HCJ_EXT, "material_name"),
                      (TC_EXT, "formula")):
        d = pd.read_csv(path, low_memory=False)
        ref_frames.append(pd.Series(d[col].astype(str).unique()))
    ref = pd.concat(ref_frames, ignore_index=True).drop_duplicates()
    Xr = featlib.build_features(ref)[FRAC_COLS].astype(float)
    Xr = Xr[Xr.notna().all(axis=1)].to_numpy(float)
    sc = StandardScaler().fit(Xr)
    Z = sc.transform(Xr)
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    ad_thr = float(np.percentile(D.min(axis=1), 95))
    print(f"  适用域阈值（训练集留一最近邻距离 95 分位）= {ad_thr:.3f}")

    payload = {"models": models, "scaler": sc, "ref": Z, "ad_thr": ad_thr,
               "features": FEATURES, "frac_cols": FRAC_COLS}
    joblib.dump(payload, CACHE)
    print(f"WROTE {CACHE}")
    return payload


def enumerate_candidates():
    """RE2(Fe1-yCoy)14B：RE = Nd/Pr/Ce/La/Dy/Tb 配比。"""
    rows = []
    for x_ce, x_la, x_hre, dy_share, y_co in itertools.product(
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.0, 0.1, 0.2],
            [0.0, 0.03, 0.06, 0.10],
            [0.5, 1.0],
            [0.0, 0.05, 0.10, 0.20]):
        if x_ce + x_la + x_hre >= 1.0:
            continue
        rest = 1.0 - x_ce - x_la - x_hre
        x_pr = 0.25 * rest          # Nd:Pr 固定按 3:1
        x_nd = 0.75 * rest
        x_dy = x_hre * dy_share
        x_tb = x_hre * (1.0 - dy_share)
        re = {"Nd": 2 * x_nd, "Pr": 2 * x_pr, "Ce": 2 * x_ce, "La": 2 * x_la,
              "Dy": 2 * x_dy, "Tb": 2 * x_tb}
        tm = {"Fe": 14 * (1 - y_co), "Co": 14 * y_co}
        comp = {k: v for k, v in {**re, **tm, "B": 1.0}.items() if v > 1e-9}
        formula = "".join(f"{k}{round(v, 4):g}" for k, v in comp.items())
        rows.append({"formula": formula, "x_Ce": x_ce, "x_La": x_la,
                     "x_HRE": x_hre, "y_Co": y_co, "Dy_share": dy_share})
    df = pd.DataFrame(rows)
    # x_HRE=0 时 Dy_share 无意义，会产生重复配方，按化学式去重
    return df.drop_duplicates(subset="formula").reset_index(drop=True)


def compromise_shortlist(sub: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """折中规划：三个目标各自归一到 [0,1] 后取到理想点的距离，升序取前 n。"""
    ms = sub["Ms_emu_g"].to_numpy(float)
    tc = sub["Tc_K"].to_numpy(float)
    cost = sub["cost_usd_per_kg"].to_numpy(float)

    def unit_benefit(v):
        rng = v.max() - v.min()
        return np.ones_like(v) if rng <= 0 else (v - v.min()) / rng

    score = np.sqrt((1 - unit_benefit(ms)) ** 2
                    + (1 - unit_benefit(tc)) ** 2
                    + unit_benefit(cost) ** 2)
    out = sub.copy()
    out["compromise_score"] = np.round(score, 4)
    return out.sort_values("compromise_score").head(n).reset_index(drop=True)


def pareto_mask(ms, tc, cost):
    n = len(ms)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if (ms[j] >= ms[i] and tc[j] >= tc[i] and cost[j] <= cost[i]
                    and (ms[j] > ms[i] or tc[j] > tc[i] or cost[j] < cost[i])):
                keep[i] = False
                break
    return keep


def main() -> int:
    payload = train_models()
    models, sc, ref, ad_thr = (payload["models"], payload["scaler"],
                               payload["ref"], payload["ad_thr"])

    prices = pd.read_csv(os.path.join(DATA_DIR, "06_element_cost_criticality.csv"),
                         encoding="utf-8-sig")
    price_map = dict(zip(prices["element"], prices["price_usd_per_kg"]))

    cand = enumerate_candidates()
    print(f"\n候选配方 {len(cand)} 个")
    F = featlib.build_features(cand["formula"], prices=price_map)
    Xc = F[FEATURES].astype(float)

    cand["Ms_emu_g"] = np.round(models["ms"].predict(Xc), 2)
    cand["Tc_K"] = np.round(models["tc"].predict(Xc), 1)
    cand["Hcj_kA_m"] = np.round(
        10 ** models["hcj"].predict(Xc) - 1.0, 1)
    cand["cost_usd_per_kg"] = np.round(F["raw_cost_index_usd_per_kg"], 2)
    cand["hre_mass_frac"] = np.round(F["hre_mass_frac"], 4)
    cand["M_g_per_mol"] = cand["formula"].map(
        lambda f: float(Composition(str(f)).weight))
    cand["Ms_muB_fu"] = np.round(cand["Ms_emu_g"] * cand["M_g_per_mol"]
                                 / MU_B_PER_EMU_G, 2)

    q = sc.transform(F[FRAC_COLS].astype(float).to_numpy())
    cand["ad_dist"] = np.round(
        np.sqrt(((q[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)).min(axis=1), 3)
    cand["in_domain"] = cand["ad_dist"] <= ad_thr

    print(f"  落在适用域内 {int(cand['in_domain'].sum())} 个")
    print(f"  Hcj 展示区间 {cand['Hcj_kA_m'].min():.0f}~{cand['Hcj_kA_m'].max():.0f} kA/m"
          "（含 Dy/Tb 的预测不可信，仅作展示）")

    cand.to_csv(os.path.join(DATA_DIR, "31_search_candidates.csv"),
                index=False, encoding="utf-8-sig")
    print(f"WROTE 31_search_candidates.csv  {len(cand)} 行")

    # ---- 多目标前沿：Ms 与 Tc 取大，成本取小 ----
    sub = cand[cand["in_domain"]].reset_index(drop=True)
    keep = pareto_mask(sub["Ms_emu_g"].to_numpy(), sub["Tc_K"].to_numpy(),
                       sub["cost_usd_per_kg"].to_numpy())
    front = sub[keep].copy()
    front["pareto_rank"] = 1
    front = front.sort_values("cost_usd_per_kg").reset_index(drop=True)
    front.to_csv(os.path.join(DATA_DIR, "31_search_pareto.csv"),
                 index=False, encoding="utf-8-sig")
    print(f"WROTE 31_search_pareto.csv  {len(front)} 个非支配配方")
    cols = ["formula", "x_Ce", "x_La", "x_HRE", "y_Co", "Ms_emu_g", "Ms_muB_fu",
            "Tc_K", "Hcj_kA_m", "cost_usd_per_kg", "hre_mass_frac"]
    print(front[cols].round(2).to_string(index=False))

    line = "=" * 76
    print("\n" + line)
    print(f"推荐短名单（折中排序，从 {len(front)} 个非支配解里取 5 个）")
    print(line)
    short = compromise_shortlist(front, 5)
    short.to_csv(os.path.join(DATA_DIR, "31_search_shortlist.csv"),
                 index=False, encoding="utf-8-sig")
    print(short[["formula", "x_Ce", "x_La", "x_HRE", "y_Co", "Ms_emu_g",
                 "Ms_muB_fu", "Tc_K", "Hcj_kA_m", "cost_usd_per_kg",
                 "compromise_score"]].round(2).to_string(index=False))

    print("\nCo 取代对 Ms 的影响（检查趋势是否单调、有无训练覆盖空洞）：")
    base = sub[(sub["x_Ce"] == 0) & (sub["x_La"] == 0) & (sub["x_HRE"] == 0)]
    base = base.sort_values("y_Co")
    for _, r in base.iterrows():
        print(f"  y_Co={r['y_Co']:.2f}  Ms={r['Ms_emu_g']:7.2f} emu/g  "
              f"Tc={r['Tc_K']:6.1f} K  成本={r['cost_usd_per_kg']:6.2f}")

    print("\nLa 取代对 Ms 的影响（实验上 La 无 4f 磁矩，应使 Ms 下降）：")
    la_series = sub[(sub["x_Ce"] == 0) & (sub["x_HRE"] == 0) & (sub["y_Co"] == 0.05)]
    la_series = la_series.sort_values("x_La")
    for _, r in la_series.iterrows():
        print(f"  x_La={r['x_La']:.2f}  Ms={r['Ms_emu_g']:7.2f} emu/g  "
              f"Tc={r['Tc_K']:6.1f} K  成本={r['cost_usd_per_kg']:6.2f}")

    # 成本最低且 Ms 最高的几个代表性端点
    best_ms = sub.loc[sub["Ms_emu_g"].idxmax()]
    best_tc = sub.loc[sub["Tc_K"].idxmax()]
    cheap = sub.loc[sub["cost_usd_per_kg"].idxmin()]
    print("\n单目标端点：")
    for tag, r in (("Ms 最高", best_ms), ("Tc 最高", best_tc), ("成本最低", cheap)):
        print(f"  {tag:8s} {r['formula']:<34} Ms={r['Ms_emu_g']:7.2f} emu/g  "
              f"Tc={r['Tc_K']:6.1f} K  成本={r['cost_usd_per_kg']:7.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib 不可用，跳过绘图：{exc}")
    else:
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=200)
        sca = ax.scatter(sub["cost_usd_per_kg"], sub["Ms_emu_g"],
                         c=sub["Tc_K"], cmap="viridis", s=26, alpha=0.75)
        ax.scatter(front["cost_usd_per_kg"], front["Ms_emu_g"], s=70,
                   facecolors="none", edgecolors="#c0392b", linewidths=1.4,
                   label="Pareto front")
        cb = fig.colorbar(sca, ax=ax)
        cb.set_label("predicted Tc (K)")
        ax.set_xlabel("raw cost index (USD/kg)")
        ax.set_ylabel("predicted Ms (emu/g)")
        ax.set_title("Composition search: max Ms & Tc, min cost (RE2(Fe,Co)14B)")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        png = os.path.join(DATA_DIR, "31_search_pareto.png")
        fig.savefig(png)
        plt.close(fig)
        print(f"WROTE {png}")

    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keys = ["31_search_candidates.csv", "31_search_pareto.csv"]
        keys.append("31_search_shortlist.csv")
        remain = idx[~idx["file"].isin(keys)]
        new = pd.DataFrame([
            {"file": "31_search_candidates.csv", "layer": "优化",
             "rows": len(cand), "columns": len(cand.columns),
             "source": "三个成分模型预测（scripts/search_compositions.py）",
             "purpose": "枚举候选配方的 Ms/Tc/Hcj 预测、成本与适用域判定",
             "refresh_note": "重跑 scripts/search_compositions.py 刷新"},
            {"file": "31_search_pareto.csv", "layer": "优化",
             "rows": len(front), "columns": len(front.columns),
             "source": "由 31 候选筛非支配（scripts/search_compositions.py）",
             "purpose": "Ms/Tc/成本三目标的非支配候选配方（Hcj 不参与判定）",
             "refresh_note": "重跑 scripts/search_compositions.py 刷新"},
            {"file": "31_search_shortlist.csv", "layer": "优化",
             "rows": len(short), "columns": len(short.columns),
             "source": "由 31 前沿折中排序（scripts/search_compositions.py）",
             "purpose": "从非支配解中按折中距离挑出的推荐配方短名单",
             "refresh_note": "重跑 scripts/search_compositions.py 刷新"},
        ])
        pd.concat([remain, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print(f"登记 {len(new)} 行 -> 00_dataset_index.csv")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
