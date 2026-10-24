"""饱和磁化强度 Ms 的成分模型：与 Tc / Hcj 同一套特征空间，供成分搜索调用。

读入
----
外部 Ms 数据  ../02_Devs/Basis_Of_AI_Teamwork/Ms/outputs/Ms_cleaned.csv
data/18_features_all.csv   设计空间（含 DFT 的 Ms_T 与密度，用于跨来源校验）

产出
----
data/28_ms_model_metrics.csv       交叉验证指标 + 跨来源校验
data/28_ms_imputed_design_space.csv 设计空间 104 行的预测 Ms（emu/g 与 μ0Ms）

口径
----
    - 目标取 Ms_emu_g（质量磁化强度，emu/g）而非 μ0Ms（T）：emu/g 是直接测量量，
      不需要密度即可换算，可用样本 3193 行；而 emu/g -> μ0Ms 需要密度，
      密度混算规则在本项目 104 条 DFT 密度上的中位误差达 10%，不足以承担这一步换算。
      μ0Ms 只在本表已知 DFT 密度的行上给出（μ0Ms[T] = Ms_emu_g * rho * 1.2566e-3）。
    - 上游 need_density（234 行）与 ambiguous（30 行）的换算状态不可用，剔除。
    - 同一材料多测量按 material_name 取中位数聚合。
    - 交叉验证同时给随机折与按化学体系分组折；并单独报 RE-Fe-B 子群。
    - 跨来源校验：本模型从未见过 DFT 数据，用它去预测设计空间 77 个化合物，
      再与 DFT 的 Ms_T 换算值比较，衡量"实验模型能否给 DFT 设计空间补标签"。
用法
----
python scripts/train_ms.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import featlib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MS_PATH = (r"C:\Users\Administrator\Desktop\02_Devs\Basis_Of_AI_Teamwork"
           r"\Ms\outputs\Ms_cleaned.csv")

TARGET = "Ms_emu_g"
EMU_G_TO_T = 1.2566e-3  # μ0Ms[T] = Ms_emu_g * rho[g/cm3] * 1.2566e-3

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

RE_SET = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
          "Er", "Tm", "Yb", "Lu", "Y", "Sc"}


def elements_of(formula) -> set:
    try:
        return {e.symbol for e in Composition(str(formula)).elements}
    except Exception:  # noqa: BLE001
        return set()


def chem_system(formula) -> str:
    els = elements_of(formula)
    return "-".join(sorted(els)) if els else "UNPARSED"


def metrics(y_true, y_pred) -> dict:
    return {"r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def line(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def main() -> int:
    if not os.path.exists(MS_PATH):
        print(f"找不到外部 Ms 数据：{MS_PATH}")
        return 1
    raw = pd.read_csv(MS_PATH, low_memory=False)
    print(f"外部 Ms 原始 {len(raw)} 行")

    line("1. 换算状态筛选")
    print("换算状态：", dict(raw["Ms_conv_status"].value_counts()))
    usable = raw[(raw["Ms_conv_status"] == "ok") & raw[TARGET].notna()].copy()
    print(f"可用 {len(usable)} 行（need_density / ambiguous 无法换算，剔除）")
    print("注：need_density 那 234 行是 T / G / kG / emu per cm3 口径，"
          "补上密度即可复算，是后续可扩的样本池。")

    line("2. 按材料聚合 + 特征抽取")
    agg = (usable.groupby("material_name", as_index=False)
           .agg(Ms_emu_g=(TARGET, "median"), n_meas=(TARGET, "size")))
    agg["chem_sys"] = agg["material_name"].map(chem_system)
    agg = agg[(agg["chem_sys"] != "UNPARSED") & agg["Ms_emu_g"].notna()].reset_index(drop=True)
    X = featlib.build_features(agg["material_name"].values)[FEATURES].astype(float)
    ok = X[RATIO_COLS].notna().all(axis=1)
    agg, X = agg[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
    y = agg["Ms_emu_g"].to_numpy(dtype=float)
    print(f"聚合后 {len(agg)} 个材料；标签中位 {np.median(y):.1f} emu/g，"
          f"范围 {y.min():.1f}~{y.max():.1f}")

    els = agg["material_name"].map(elements_of)
    agg["is_refeb"] = els.map(lambda s: bool(s & RE_SET) and {"Fe", "B"} <= s)
    print(f"RE-Fe-B 子群 {int(agg['is_refeb'].sum())} 行；"
          f"化学体系数 {agg['chem_sys'].nunique()}")

    line("3. 交叉验证：随机 5 折 vs 按化学体系分组 5 折")
    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "GBM": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                         max_depth=4, min_samples_leaf=2,
                                         random_state=0),
        "RF": RandomForestRegressor(n_estimators=800, min_samples_leaf=2,
                                    random_state=0),
    }
    rows, preds = [], {}
    for name, model in models.items():
        for cv_name, cv, groups in [
            ("kfold", KFold(5, shuffle=True, random_state=0), None),
            ("chemsys", GroupKFold(5), agg["chem_sys"].to_numpy()),
        ]:
            Xf = X.copy()
            if name == "ridge":
                Xf = pd.DataFrame(StandardScaler().fit_transform(Xf), columns=Xf.columns)
            p = cross_val_predict(model, Xf, y, cv=cv, groups=groups)
            preds[(name, cv_name)] = p
            m = metrics(y, p)
            rows.append({"model": name, "cv": cv_name, "n_train": len(X), **m,
                         "unit": "emu/g", "note": ""})
            print(f"{name:12s} {cv_name:8s} R2={m['r2']:.3f}  MAE={m['mae']:6.2f}  "
                  f"RMSE={m['rmse']:6.2f} emu/g")

    line("4. 子群误差（GBM，分组折）")
    pg = preds[("GBM", "chemsys")]
    m_refeb = metrics(y[agg["is_refeb"].to_numpy()], pg[agg["is_refeb"].to_numpy()])
    rows.append({"model": "GBM", "cv": "chemsys::RE-Fe-B",
                 "n_train": int(agg["is_refeb"].sum()), **m_refeb,
                 "unit": "emu/g", "note": "子群误差"})
    print(f"RE-Fe-B 子群 n={int(agg['is_refeb'].sum())}  R2={m_refeb['r2']:.3f}  "
          f"MAE={m_refeb['mae']:.2f} emu/g")

    line("5. 跨来源校验：用实验模型去预测 DFT 设计空间")
    feat = pd.read_csv(os.path.join(DATA_DIR, "18_features_all.csv"), low_memory=False)
    ds = feat[(feat["pool"] == "design_space")
              & (feat["ordering"].isin(["FM", "FiM"]))
              & (feat["flagged_exotic"] == 0)].copy()
    X_ds = featlib.build_features(ds["formula_pretty"].values)[FEATURES].astype(float)
    final = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                      max_depth=4, min_samples_leaf=2,
                                      random_state=0).fit(X, y)
    print(f"全量训练 R2={r2_score(y, final.predict(X)):.3f}（样本内，仅供参考）")
    ms_emu_g = final.predict(X_ds)
    rho = ds["density_g_cm3"].to_numpy(dtype=float)
    mu0_ms = ms_emu_g * rho * EMU_G_TO_T
    dft = ds["Ms_T"].to_numpy(dtype=float)
    cmp = pd.DataFrame({"formula_pretty": ds["formula_pretty"].values,
                        "dft_Ms_T": dft, "pred_mu0Ms_T": mu0_ms,
                        "pred_Ms_emu_g": ms_emu_g, "density": rho})
    cmp["gap_T"] = cmp["pred_mu0Ms_T"] - cmp["dft_Ms_T"]
    print(f"设计空间 {len(cmp)} 行；与 DFT Ms_T 的相关系数 "
          f"{np.corrcoef(cmp['dft_Ms_T'], cmp['pred_mu0Ms_T'])[0, 1]:.3f}")
    print(f"平均偏差 {cmp['gap_T'].mean():+.3f} T（正=实验模型偏高），"
          f"MAE {cmp['gap_T'].abs().mean():.3f} T")
    print(cmp.reindex(cmp["dft_Ms_T"].sort_values(ascending=False).index)
          .head(12).round(3).to_string(index=False))
    m_x = metrics(cmp["dft_Ms_T"], cmp["pred_mu0Ms_T"])
    rows.append({"model": "GBM", "cv": "cross-source::design_space",
                 "n_train": len(cmp), "r2": m_x["r2"], "mae": m_x["mae"],
                 "rmse": m_x["rmse"], "unit": "T",
                 "note": "实验训练模型 vs DFT Ms_T（跨来源）"})

    line("6. 保存产出并登记")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(DATA_DIR, "28_ms_model_metrics.csv"),
                      index=False, encoding="utf-8-sig")
    print(metrics_df.round(3).to_string(index=False))

    out = ds[["material_id", "formula_pretty", "chemsys", "ordering", "Ms_T",
              "density_g_cm3", "is_stable", "energy_above_hull_eV",
              "raw_cost_index_usd_per_kg"]].copy()
    out["ms_emu_g_pred"] = np.round(ms_emu_g, 2)
    out["mu0_Ms_T_pred"] = np.round(mu0_ms, 4)
    out["mu0_Ms_T_dft"] = out["Ms_T"]
    out["gap_vs_dft_T"] = np.round(cmp["gap_T"].to_numpy(), 4)
    out = out.sort_values("mu0_Ms_T_pred", ascending=False).reset_index(drop=True)
    out.to_csv(os.path.join(DATA_DIR, "28_ms_imputed_design_space.csv"),
               index=False, encoding="utf-8-sig")
    print(f"WROTE 28_ms_imputed_design_space.csv  {len(out)} 行")

    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keys = ["28_ms_model_metrics.csv", "28_ms_imputed_design_space.csv"]
        keep = idx[~idx["file"].isin(keys)]
        new = pd.DataFrame([
            {"file": "28_ms_model_metrics.csv", "layer": "建模",
             "rows": len(metrics_df), "columns": len(metrics_df.columns),
             "source": "外部 Ms 数据 + 18（scripts/train_ms.py）",
             "purpose": "成分法 Ms 模型的交叉验证、子群误差与 DFT 跨来源校验",
             "refresh_note": "重跑 scripts/train_ms.py 刷新"},
            {"file": "28_ms_imputed_design_space.csv", "layer": "标签",
             "rows": len(out), "columns": len(out.columns),
             "source": "18 + 模型预测（scripts/train_ms.py）",
             "purpose": "设计空间的成分法 Ms 预测（emu/g 与 μ0Ms）及与 DFT 的偏差",
             "refresh_note": "重跑 scripts/train_ms.py 刷新"},
        ])
        pd.concat([keep, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print("登记 2 行 -> 00_dataset_index.csv")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
