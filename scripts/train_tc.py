"""居里温度模型：用外部 Tc 数据训练，给设计空间里没实验值的组分补标签。

读入
----
外部 Tc 数据  ../02_Devs/Basis_Of_AI_Teamwork/Tc/tc_repm_deliverables/data/tc_dataset.csv
data/22_tc_join_design_space.csv  已匹配的实验 Tc 中位数（用于真实留出检验）
data/18_features_all.csv           设计空间的 Ms/成本/稳定性（用于合并输出）

产出
----
data/23_tc_model_metrics.csv       交叉验证 + 设计空间 23 行真实留出检验的指标
data/24_tc_imputed_design_space.csv 设计空间 104 行：实验 Tc + 模型补全 Tc + 最终 Tc

口径
----
    - 训练集：外部 Tc 数据按化学式取中位数（去重），剔除 6 条已知错误值与 tc_K<10 的
      低值哨兵，再剔除无稀土/特征缺失的少量行。
    - 特征只取"化学式能推出的"成分描述符（原子分数/比值/加权量/磁性先验），
      训练与预测完全同构，避免用 DFT 专属量（密度/凸包能）造成不可迁移。
    - 检验用设计空间那 23 个有实验 Tc 的组分做真实留出：模型只见过外部数据，
      没见过它们，这 23 行的误差就是补全 81 行时可信度的直接量度。
用法
----
python scripts/train_tc.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import featlib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TC_PATH = r"C:\Users\Administrator\Desktop\02_Devs\Basis_Of_AI_Teamwork\Tc\tc_repm_deliverables\data\tc_dataset.csv"

FLAGGED = {
    ("Nd2Fe14B", 450.0), ("Nd2Fe14B", 768.15), ("Nd2Fe17", 743.0),
    ("Ce2Fe14B", 490.0), ("Er2Fe14B", 314.0), ("Y2Co17", 940.0),
}

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


def metrics(y_true, y_pred):
    return {"r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def line(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    if not os.path.exists(TC_PATH):
        print(f"找不到外部 Tc 数据：{TC_PATH}")
        return 1
    tc = pd.read_csv(TC_PATH, low_memory=False)
    tc = tc[~tc.apply(lambda r: (str(r["formula"]), float(r["tc_K"])) in FLAGGED,
                      axis=1)]
    tc = tc[tc["tc_K"] >= 10].copy()  # 剔除低值哨兵
    train_agg = (tc.groupby("formula", as_index=False)
                 .agg(tc_K=("tc_K", "median"), n_sources=("tc_K", "size")))
    print(f"外部 Tc：{len(tc)} 行 -> 按化学式聚合为 {len(train_agg)} 个训练样本")

    X = featlib.build_features(train_agg["formula"].values)
    X = X[FEATURES].astype(float)
    y = train_agg["tc_K"].to_numpy(dtype=float)
    keep = X[["b_over_re", "fe_over_re", "co_over_tm", "hre_over_re",
              "lre_over_re"]].notna().all(axis=1)
    X, y = X[keep], y[keep]
    print(f"特征可用样本：{len(X)}（其余为无稀土或成分解析失败，已剔除）")

    line("1. 交叉验证（5 折，按化学式去重后的样本上）")
    rows = []
    for name, model in {
        "GBM": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                         max_depth=4, min_samples_leaf=2,
                                         random_state=0),
        "RF": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2,
                                    random_state=0),
    }.items():
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        r2s, maes, rmses = [], [], []
        for tr, va in kf.split(X):
            m = model
            m.fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[va])
            r2s.append(r2_score(y[va], p))
            maes.append(mean_absolute_error(y[va], p))
            rmses.append(np.sqrt(mean_squared_error(y[va], p)))
        rows.append({"model": name, "cv_r2_mean": float(np.mean(r2s)),
                     "cv_r2_std": float(np.std(r2s)),
                     "cv_mae_K": float(np.mean(maes)),
                     "cv_rmse_K": float(np.mean(rmses)),
                     "n_train": len(X), "n_holdout23": np.nan,
                     "holdout23_r2": np.nan, "holdout23_mae_K": np.nan,
                     "holdout23_rmse_K": np.nan,
                     "note": "外部 Tc 数据交叉验证"})
        print(f"{name}: CV R2={rows[-1]['cv_r2_mean']:.3f}±{rows[-1]['cv_r2_std']:.3f}, "
              f"MAE={rows[-1]['cv_mae_K']:.1f} K, RMSE={rows[-1]['cv_rmse_K']:.1f} K")

    line("2. 真实留出检验：设计空间 23 个有实验 Tc 的组分")
    join_path = os.path.join(DATA_DIR, "22_tc_join_design_space.csv")
    if not os.path.exists(join_path):
        print(f"找不到 {join_path}，先跑 scripts/join_tc.py")
        return 1
    join = pd.read_csv(join_path, low_memory=False)
    held = join[join["tc_matched"]].copy()
    X_held = featlib.build_features(held["formula_pretty"].values)[FEATURES].astype(float)
    y_held = held["tc_K_median"].to_numpy(dtype=float)

    gbm = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                    max_depth=4, min_samples_leaf=2,
                                    random_state=0).fit(X, y)
    pred_held = gbm.predict(X_held)
    m_held = metrics(y_held, pred_held)
    print(f"GBM 在 23 行留出集：R2={m_held['r2']:.3f}, MAE={m_held['mae']:.1f} K, "
          f"RMSE={m_held['rmse']:.1f} K")
    cmp = pd.DataFrame({
        "formula_pretty": held["formula_pretty"].values,
        "tc_experimental": y_held,
        "tc_predicted": pred_held,
    })
    cmp["abs_err_K"] = (cmp["tc_predicted"] - cmp["tc_experimental"]).abs()
    cmp = cmp.sort_values("abs_err_K", ascending=False)
    print(cmp.round(1).to_string(index=False))
    rows.append({"model": "GBM", "cv_r2_mean": np.nan, "cv_r2_std": np.nan,
                 "cv_mae_K": np.nan, "cv_rmse_K": np.nan, "n_train": len(X),
                 "n_holdout23": int(len(held)),
                 "holdout23_r2": m_held["r2"], "holdout23_mae_K": m_held["mae"],
                 "holdout23_rmse_K": m_held["rmse"],
                 "note": "GBM 在设计空间 23 个有实验 Tc 组分的真实留出"})

    line("3. 补全设计空间 104 行的 Tc")
    feat = pd.read_csv(os.path.join(DATA_DIR, "18_features_all.csv"), low_memory=False)
    ds = feat[feat["pool"] == "design_space"].copy()
    X_ds = featlib.build_features(ds["formula_pretty"].values)[FEATURES].astype(float)
    pred_all = gbm.predict(X_ds)
    ds = ds.merge(join[["material_id", "tc_K_median", "tc_n_sources"]],
                  on="material_id", how="left")
    out = ds[["material_id", "formula_pretty", "chemsys", "ordering", "Ms_T",
              "is_stable", "energy_above_hull_eV", "raw_cost_index_usd_per_kg"]].copy()
    out["tc_experimental_median"] = ds["tc_K_median"]
    out["tc_predicted"] = pred_all.round(1)
    out["tc_final"] = ds["tc_K_median"].fillna(pd.Series(pred_all, index=ds.index)).round(1)
    out["tc_source"] = np.where(ds["tc_K_median"].notna(), "experimental", "predicted")
    out["tc_n_sources"] = ds["tc_n_sources"]
    out = out.sort_values("Ms_T", ascending=False).reset_index(drop=True)
    out.to_csv(os.path.join(DATA_DIR, "24_tc_imputed_design_space.csv"),
               index=False, encoding="utf-8-sig")
    print(f"WROTE 24_tc_imputed_design_space.csv  实验 {int((out['tc_source']=='experimental').sum())} 行，"
          f"补全 {int((out['tc_source']=='predicted').sum())} 行")

    line("4. 保存指标并登记")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(DATA_DIR, "23_tc_model_metrics.csv"),
                      index=False, encoding="utf-8-sig")
    print(metrics_df.round(3).to_string(index=False))

    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keep_idx = idx[~idx["file"].isin(["23_tc_model_metrics.csv",
                                          "24_tc_imputed_design_space.csv"])]
        new = pd.DataFrame([
            {"file": "23_tc_model_metrics.csv", "layer": "建模", "rows": len(metrics_df),
             "columns": len(metrics_df.columns),
             "source": "外部 Tc + 22（scripts/train_tc.py）",
             "purpose": "Tc 模型交叉验证与设计空间真实留出检验指标",
             "refresh_note": "重跑 scripts/train_tc.py 刷新"},
            {"file": "24_tc_imputed_design_space.csv", "layer": "标签",
             "rows": len(out), "columns": len(out.columns),
             "source": "22 的实验 Tc + 模型补全（scripts/train_tc.py）",
             "purpose": "设计空间每行带实验/补全后的最终 Tc 及来源标记",
             "refresh_note": "重跑 scripts/train_tc.py 刷新"},
        ])
        pd.concat([keep_idx, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print("登记 -> 00_dataset_index.csv")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
