"""Ms_T 可解释基线：验证特征工程的信号，而非追求最高分数。

读入
----
data/18_features_all.csv  特征矩阵（scripts/feature_engineering.py 产出）

产出
----
data/19_ms_baseline_metrics.csv        各模型在冻结切分上的 R2/MAE/RMSE
data/20_ms_feature_importance.csv      特征的相关性 + 增益重要性 + 置换重要性

口径
----
    - 只用设计空间（pool=design_space），且 ordering 取 FM/FiM、flagged_exotic=0。
      这对应"去掉硼酸盐/氧化物与成分不闭合的金属间化合物"，是最干净的一档。
    - 严格复用冻结的 split，标准化等一律只在 train 上拟合（这里用树模型，不涉及）。
    - Ms_emu_g / moment_uB_per_fu 是与 Ms_T 同源或近亲的列，明确排除，防止泄漏。
用法
----
python scripts/baseline_ms.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

TARGET = "Ms_T"

# 成分分数（原子 + 质量）与"未跟踪元素"占比
TRACKED = ["Nd", "Pr", "Dy", "Tb", "Ce", "Sm", "Gd", "Y", "Ho", "Er", "La", "Lu",
           "Co", "Cu", "Ga", "Al", "Si", "V", "Ti", "Nb", "Zr", "Mn", "Ni",
           "Cr", "Fe", "B"]
FRAC_COLS = [f"frac_{e}" for e in TRACKED] + ["frac_other"]
RATIO_COLS = ["re_frac", "tm_frac", "b_frac", "re_over_metal", "b_over_re",
              "fe_over_re", "co_over_tm", "hre_over_re", "lre_over_re"]
WEIGHTED_COLS = ["w_mean_z", "w_mean_mass", "w_mean_radius", "w_mean_electroneg",
                 "w_mean_valence", "mix_entropy", "size_mismatch",
                 "electroneg_spread"]
MAGNETIC_COLS = ["re_avg_degennes", "re_avg_moment", "tm_avg_moment",
                 "has_re", "has_hre"]
COST_COLS = ["hre_mass_frac", "raw_cost_index_usd_per_kg"]
STRUCT_COLS = ["density_g_cm3", "n_elements", "n_sites"]


def build_x(df: pd.DataFrame, fill_na: bool = False,
            columns: list | None = None) -> pd.DataFrame:
    """构造特征矩阵：数值特征 + 晶系 one-hot。columns 给定时按它对齐。"""
    cols = FRAC_COLS + RATIO_COLS + WEIGHTED_COLS + MAGNETIC_COLS + COST_COLS + STRUCT_COLS
    x = df[[c for c in cols if c in df.columns]].copy()
    if "crystal_system" in df.columns:
        cs = pd.get_dummies(df["crystal_system"], prefix="crystal").astype(float)
        x = pd.concat([x.reset_index(drop=True), cs.reset_index(drop=True)], axis=1)
    if fill_na:
        x = x.fillna(0.0)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0.0)
    return x


def metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def line(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    feat_path = os.path.join(DATA_DIR, "18_features_all.csv")
    if not os.path.exists(feat_path):
        print(f"找不到 {feat_path}，先跑 scripts/feature_engineering.py")
        return 1
    all_df = pd.read_csv(feat_path, low_memory=False)

    ds = all_df[all_df["pool"] == "design_space"].copy()
    usable = ds[(ds["ordering"].isin(["FM", "FiM"])) & (ds["flagged_exotic"] == 0)].copy()
    print(f"设计空间 {len(ds)} 行 -> 可用 {len(usable)} 行（FM/FiM 且非氧化物/非成分不闭合）")
    print("  可用集切分：" + str({k: int(v) for k, v in usable["split"].value_counts().items()}))

    train = usable[usable["split"] == "train"]
    val = usable[usable["split"] == "val"]
    test = usable[usable["split"] == "test"]

    x_all_cols = build_x(usable).columns
    x_train = build_x(train, columns=x_all_cols)
    x_val = build_x(val, columns=x_all_cols)
    x_test = build_x(test, columns=x_all_cols)
    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    line("1. 信号检查：各特征与 Ms_T 的 Spearman 相关（训练集）")
    corr = {}
    for c in x_train.columns:
        if x_train[c].nunique() > 1:
            rho = spearmanr(x_train[c], y_train).correlation
            corr[c] = 0.0 if np.isnan(rho) else float(rho)
    corr_s = pd.Series(corr).sort_values(key=lambda s: s.abs(), ascending=False)
    print(corr_s.head(15).to_string())

    line("2. 模型对比（冻结切分）")
    rows = []

    dummy = DummyRegressor(strategy="mean").fit(x_train, y_train)
    rows.append({"model": "dummy_mean", "n_train": len(train), "n_val": len(val),
                 "n_test": len(test),
                 "r2_train": float(r2_score(y_train, dummy.predict(x_train))),
                 "r2_val": float(r2_score(y_val, dummy.predict(x_val))),
                 "r2_test": float(r2_score(y_test, dummy.predict(x_test))),
                 "mae_test": metrics(y_test, dummy.predict(x_test))["mae"],
                 "rmse_test": metrics(y_test, dummy.predict(x_test))["rmse"],
                 "note": "预测训练集均值"})

    models = {
        "GBM_depth3": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            min_samples_leaf=2, random_state=0),
        "RF": RandomForestRegressor(
            n_estimators=800, max_features="sqrt", min_samples_leaf=2,
            random_state=0),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        rows.append({"model": name, "n_train": len(train), "n_val": len(val),
                     "n_test": len(test),
                     "r2_train": float(r2_score(y_train, model.predict(x_train))),
                     "r2_val": float(r2_score(y_val, model.predict(x_val))),
                     "r2_test": float(r2_score(y_test, model.predict(x_test))),
                     "mae_test": metrics(y_test, model.predict(x_test))["mae"],
                     "rmse_test": metrics(y_test, model.predict(x_test))["rmse"],
                     "note": "设计空间 FM/FiM 清洗子集"})

    # 迁移对照：在背景池（FM/FiM，非氧化物）上训练，评估设计空间测试集
    ctx = all_df[(all_df["pool"] == "context") & (all_df["ordering"].isin(["FM", "FiM"]))
                 & (all_df["flagged_exotic"] == 0)].copy()
    if len(ctx) > 100:
        x_ctx = build_x(ctx, fill_na=True, columns=x_all_cols)
        x_ctx, y_ctx = x_ctx, ctx[TARGET].fillna(ctx[TARGET].median())
        cols = x_ctx.columns.intersection(x_test.columns)
        gbm_ctx = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            min_samples_leaf=5, random_state=0).fit(x_ctx[cols], y_ctx)
        y_pred = gbm_ctx.predict(x_test[cols])
        rows.append({"model": "GBM_pretrain_context", "n_train": len(ctx),
                     "n_val": len(val), "n_test": len(test),
                     "r2_train": float(r2_score(y_ctx, gbm_ctx.predict(x_ctx[cols]))),
                     "r2_val": float(r2_score(y_val, gbm_ctx.predict(x_val[cols]))),
                     "r2_test": float(r2_score(y_test, y_pred)),
                     "mae_test": metrics(y_test, y_pred)["mae"],
                     "rmse_test": metrics(y_test, y_pred)["rmse"],
                     "note": "背景池 FM/FiM 预训练，直接评估设计空间测试集"})
        print(f"背景池可用 {len(ctx)} 行，作迁移对照。")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(DATA_DIR, "19_ms_baseline_metrics.csv"),
                      index=False, encoding="utf-8-sig")
    print(metrics_df.round(3).to_string(index=False))

    line("3. 特征重要性（GBM 增益 + 测试集置换重要性）")
    gbm = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        min_samples_leaf=2, random_state=0).fit(x_train, y_train)
    gain = pd.Series(gbm.feature_importances_, index=x_train.columns)
    perm = permutation_importance(
        gbm, x_test, y_test, n_repeats=30, random_state=0, scoring="neg_mean_absolute_error")
    perm_s = pd.Series(perm.importances_mean, index=x_test.columns)
    imp = pd.DataFrame({
        "feature": x_train.columns,
        "spearman_train": [corr.get(c, np.nan) for c in x_train.columns],
        "importance_gain": gain,
        "importance_perm_test": perm_s,
    }).sort_values("importance_perm_test", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(DATA_DIR, "20_ms_feature_importance.csv"),
               index=False, encoding="utf-8-sig")
    print(imp.head(15).to_string(index=False))

    line("4. 登记到数据清单与字典")
    index_rows = [
        {"file": "19_ms_baseline_metrics.csv", "layer": "建模", "rows": len(metrics_df),
         "columns": len(metrics_df.columns),
         "source": "由 18 派生（scripts/baseline_ms.py）",
         "purpose": "Ms_T 基线各模型在冻结切分上的 R2/MAE/RMSE（含 dummy 与迁移对照）",
         "refresh_note": "重跑 scripts/baseline_ms.py 刷新"},
        {"file": "20_ms_feature_importance.csv", "layer": "建模", "rows": len(imp),
         "columns": len(imp.columns),
         "source": "由 18 派生（scripts/baseline_ms.py）",
         "purpose": "各特征的 Spearman 相关、增益重要性与置换重要性",
         "refresh_note": "重跑 scripts/baseline_ms.py 刷新"},
    ]
    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keys = {r["file"] for r in index_rows}
        keep = idx[~idx["file"].isin(keys)]
        pd.concat([keep, pd.DataFrame(index_rows)], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print(f"登记 {len(index_rows)} 行 -> 00_dataset_index.csv")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
