"""内禀矫顽力 Hcj 模型：用外部稀土实验数据训练，给设计空间补标签。

读入
----
外部 Hcj 数据  ../02_Devs/Basis_Of_AI_Teamwork/Hcj/outputs/Hcj_cleaned.csv
data/18_features_all.csv              设计空间（Ms/Tc/成本/稳定性）

产出
----
data/26_hcj_model_metrics.csv          各模型交叉验证 + 子群误差
data/27_hcj_imputed_design_space.csv   设计空间 104 行的 Hcj 预测 + 适用域标记

口径
----
    - 标签质量按上游 QC 的 hcj_value_class 分档取舍：
        plausible_hcj（<=20 kOe）  稀土永磁真实矫顽力量级 -> 主模型训练集
        borderline（20~60 kOe）    过渡带，主模型剔除，另做纳入对照
        likely_anisotropy_field（>60 kOe）实为各向异性场 Ha -> 一律剔除
    - 目标取 log10(1 + Hcj_kA_m)：矫顽力跨约三个数量级，原始尺度下平方误差
      会被少数大值主导，取对数后残差才有可比性。
    - 同一材料的多次测量按 material_name 取中位数聚合（与 Tc 模型同口径）。
    - 特征只用化学式可推出的成分描述符（scripts/featlib.py），与 Ms/Tc 共用同一特征空间。
      矫顽力本质是外禀性质（受晶粒尺寸、晶界扩散、退火支配），成分特征只能捕捉
      其中的内禀倾向（各向异性场量级），这一点在报告指标时必须写明。
    - 诚实检验：随机 5 折与按化学体系分组 5 折都报；并单独报 RE-Fe-B 子群误差。

已知局限（读数时必须一起看）
----
    1. 口径错位：外部 Hcj 数据来自真实磁体（含晶界相、织构、工艺），而设计空间是
       匀质单相组分。同一个 Nd2Fe14B，实验档中位约 917 kA/m，而模型对化学计量单相
       报 626 kA/m —— 这个差既是模型误差，也是"磁体 vs 化合物"的定义差。
    2. 外禀性：矫顽力受晶粒尺寸/晶界扩散/退火支配，成分特征只能捕捉内禀倾向，
       这是子群误差约 40% 的根本原因，不是换模型能解决的。
    3. 外推风险：训练集中没有 Dy2Fe14B / Tb2Fe14B 这类纯重稀土 2:14:1 块体的实验值，
       模型对它们的排序（低于 Nd2Fe14B）与"Dy/Tb 提升各向异性"的已知规律相反，
       因此当前版本不可用于评估 Dy/Tb 取代量。
用法
----
python scripts/train_hcj.py
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
HCJ_PATH = (r"C:\Users\Administrator\Desktop\02_Devs\Basis_Of_AI_Teamwork"
            r"\Hcj\outputs\Hcj_cleaned.csv")

TARGET_RAW = "Hcj_kA_m"

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

# 空间群 -> 已知硬磁家族（用于判断矫顽力预测是否落在有实验基础的体系上）
SG_FAMILY = {
    "P4_2/mnm": "2:14:1",
    "R-3m": "2:17",
    "I4/mmm": "1:12",
    "P6/mmm": "1:5",
}


def to_log(y_kA_m: np.ndarray) -> np.ndarray:
    return np.log10(1.0 + np.clip(y_kA_m, 0.0, None))


def from_log(z: np.ndarray) -> np.ndarray:
    return np.power(10.0, z) - 1.0


def elements_of(formula: str) -> set:
    try:
        return {e.symbol for e in Composition(str(formula)).elements}
    except Exception:  # noqa: BLE001
        return set()


def chem_system(formula: str) -> str:
    els = elements_of(formula)
    return "-".join(sorted(els)) if els else "UNPARSED"


def metrics(y_true_kA_m, y_pred_kA_m) -> dict:
    """在原始尺度（kA/m）与对数尺度上同时给指标。"""
    t, p = np.asarray(y_true_kA_m, float), np.asarray(y_pred_kA_m, float)
    zt, zp = to_log(t), to_log(p)
    return {
        "r2_log": float(r2_score(zt, zp)),
        "mae_log_dex": float(mean_absolute_error(zt, zp)),
        "r2_raw": float(r2_score(t, p)),
        "mae_raw_kA_m": float(mean_absolute_error(t, p)),
        "rmse_raw_kA_m": float(np.sqrt(mean_squared_error(t, p))),
        "median_abs_err_ratio": float(np.median(
            np.abs(p - t) / np.clip(t, 1.0, None))),
    }


def line(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def main() -> int:
    if not os.path.exists(HCJ_PATH):
        print(f"找不到外部 Hcj 数据：{HCJ_PATH}")
        return 1
    raw = pd.read_csv(HCJ_PATH, low_memory=False)
    print(f"外部 Hcj 原始 {len(raw)} 行")

    line("1. 标签质量分档与筛选")
    cls = raw["hcj_value_class"].value_counts().to_dict()
    print("分档构成：", cls)
    usable = raw[raw["hcj_value_class"] == "plausible_hcj"].copy()
    border = raw[raw["hcj_value_class"] == "borderline"].copy()
    print(f"主模型训练档 plausible_hcj {len(usable)} 行；"
          f"borderline {len(border)} 行（仅作敏感性对照）；"
          f"likely_anisotropy_field {int(cls.get('likely_anisotropy_field', 0))} 行剔除")

    nan_temp = int(usable["meas_temp_K"].isna().sum())
    print(f"注：测量温度未知的行 {nan_temp}/{len(usable)}，矫顽力对温度敏感，"
          "这是无法控制的标签噪声来源。")

    line("2. 按材料聚合 + 特征抽取")
    agg = (usable.groupby("material_name", as_index=False)
           .agg(hcj_kA_m=("Hcj_kA_m", "median"),
                n_meas=("Hcj_kA_m", "size"),
                tc_K=("Tc_K", "median")))
    agg = agg[agg["hcj_kA_m"].notna()].copy()
    agg["chem_sys"] = agg["material_name"].map(chem_system)
    agg = agg[agg["chem_sys"] != "UNPARSED"].copy()
    print(f"聚合后 {len(agg)} 个材料（多测量取中位数）")

    X = featlib.build_features(agg["material_name"].values)[FEATURES].astype(float)
    ok = X[RATIO_COLS].notna().all(axis=1) & X[["frac_Fe"]].notna().all(axis=1)
    agg, X = agg[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
    print(f"特征可用样本 {len(X)}")

    els = agg["material_name"].map(elements_of)
    agg["is_refeb"] = els.map(lambda s: bool(s & RE_SET) and {"Fe", "B"} <= s)
    agg["is_fere"] = els.map(lambda s: bool(s & RE_SET) and "Fe" in s)
    y_raw = agg["hcj_kA_m"].to_numpy(dtype=float)
    z = to_log(y_raw)
    print(f"标签：中位数 {np.median(y_raw):.1f} kA/m，"
          f"log10 范围 {z.min():.2f}~{z.max():.2f}")
    print(f"子群：RE-Fe-B {int(agg['is_refeb'].sum())} 行，"
          f"含 Fe 的稀土化合物 {int(agg['is_fere'].sum())} 行")
    print(f"化学体系数 {agg['chem_sys'].nunique()}")

    line("3. 交叉验证：随机 5 折 vs 按化学体系分组 5 折")
    models = {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=1.0),
        "GBM": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                         max_depth=4, min_samples_leaf=2,
                                         random_state=0),
        "RF": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2,
                                    random_state=0),
    }
    rows = []
    preds = {}
    for name, model in models.items():
        for cv_name, cv, groups in [
            ("kfold", KFold(5, shuffle=True, random_state=0), None),
            ("chemsys", GroupKFold(5), agg["chem_sys"].to_numpy()),
        ]:
            Xf = X.copy()
            if name == "ridge":
                Xf = pd.DataFrame(StandardScaler().fit_transform(Xf),
                                  columns=Xf.columns)
            p = cross_val_predict(model, Xf, z, cv=cv, groups=groups)
            preds[(name, cv_name)] = p
            m = metrics(y_raw, from_log(p))
            rows.append({"model": name, "cv": cv_name, "n_train": len(X),
                         **m, "note": ""})
            print(f"{name:14s} {cv_name:8s} R2(log)={m['r2_log']:.3f}  "
                  f"MAE={m['mae_log_dex']:.3f} dex  "
                  f"MAE={m['mae_raw_kA_m']:8.1f} kA/m  "
                  f"RMSE={m['rmse_raw_kA_m']:8.1f}")

    line("4. 子群误差：RE-Fe-B 与含 Fe 稀土化合物（GBM，分组折）")
    pg = preds[("GBM", "chemsys")]
    for tag, mask in [("RE-Fe-B", agg["is_refeb"].to_numpy()),
                      ("含 Fe 稀土", agg["is_fere"].to_numpy())]:
        m = metrics(y_raw[mask], from_log(pg[mask]))
        rows.append({"model": "GBM", "cv": f"chemsys::{tag}", "n_train": int(mask.sum()),
                     **m, "note": "子群误差"})
        print(f"{tag:12s} n={int(mask.sum()):4d}  R2(log)={m['r2_log']:.3f}  "
              f"MAE={m['mae_log_dex']:.3f} dex  MAE={m['mae_raw_kA_m']:.1f} kA/m  "
              f"中位相对误差={m['median_abs_err_ratio']:.1%}")

    line("5. 敏感性对照：把 borderline 档并入训练集")
    both = pd.concat([usable, border], ignore_index=True)
    agg2 = (both.groupby("material_name", as_index=False)
            .agg(hcj_kA_m=("Hcj_kA_m", "median")))
    agg2["chem_sys"] = agg2["material_name"].map(chem_system)
    agg2 = agg2[(agg2["chem_sys"] != "UNPARSED") & agg2["hcj_kA_m"].notna()]
    X2 = featlib.build_features(agg2["material_name"].values)[FEATURES].astype(float)
    ok2 = X2[RATIO_COLS].notna().all(axis=1)
    agg2, X2 = agg2[ok2].reset_index(drop=True), X2[ok2].reset_index(drop=True)
    z2 = to_log(agg2["hcj_kA_m"].to_numpy(dtype=float))
    gbm2 = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                     max_depth=4, min_samples_leaf=2, random_state=0)
    p2 = cross_val_predict(gbm2, X2, z2, cv=GroupKFold(5),
                           groups=agg2["chem_sys"].to_numpy())
    m2 = metrics(agg2["hcj_kA_m"].to_numpy(float), from_log(p2))
    rows.append({"model": "GBM", "cv": "chemsys::incl_borderline",
                 "n_train": len(X2), **m2, "note": "敏感性对照"})
    print(f"纳入 borderline 后 n={len(X2)}  R2(log)={m2['r2_log']:.3f}  "
          f"MAE={m2['mae_log_dex']:.3f} dex  MAE={m2['mae_raw_kA_m']:.1f} kA/m")
    print(f"（主口径 n={len(X)}：R2(log)="
          f"{metrics(y_raw, from_log(pg))['r2_log']:.3f}）")

    line("5b. 通用模型 vs 专用模型（RE-Fe-B 子群上同口径比较）")
    refeb = agg["is_refeb"].to_numpy()
    gs_refeb = agg["chem_sys"].to_numpy()[refeb]
    n_split = int(min(5, len(set(gs_refeb))))
    spec = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                     max_depth=4, min_samples_leaf=2, random_state=0)
    p_spec = cross_val_predict(spec, X[refeb], z[refeb], cv=GroupKFold(n_split),
                               groups=gs_refeb)
    m_spec = metrics(y_raw[refeb], from_log(p_spec))
    m_gen = metrics(y_raw[refeb], from_log(pg[refeb]))
    print(f"通用模型（974 行训练）在 RE-Fe-B 上  R2(log)={m_gen['r2_log']:.3f}  "
          f"MAE={m_gen['mae_log_dex']:.3f} dex  中位相对误差="
          f"{m_gen['median_abs_err_ratio']:.1%}")
    print(f"专用模型（仅 152 行训练）在 RE-Fe-B 上 R2(log)={m_spec['r2_log']:.3f}  "
          f"MAE={m_spec['mae_log_dex']:.3f} dex  中位相对误差="
          f"{m_spec['median_abs_err_ratio']:.1%}")
    rows.append({"model": "GBM_specialized", "cv": f"chemsys::{n_split}fold::RE-Fe-B",
                 "n_train": int(refeb.sum()), **m_spec,
                 "note": "只用 RE-Fe-B 子集训练"})

    line("6. 最终模型训练与设计空间补全")
    final = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                      max_depth=4, min_samples_leaf=2,
                                      random_state=0).fit(X, z)
    print(f"全量训练 R2(log)={r2_score(z, final.predict(X)):.3f}")

    feat = pd.read_csv(os.path.join(DATA_DIR, "18_features_all.csv"), low_memory=False)
    ds = feat[feat["pool"] == "design_space"].copy()
    ds["in_train"] = ds["formula_pretty"].isin(set(agg["material_name"]))
    X_ds = featlib.build_features(ds["formula_pretty"].values)[FEATURES].astype(float)
    pred_ds = from_log(final.predict(X_ds))

    # 适用域：以训练集自身留一最近邻距离的 95 分位为阈值（只用成分分数，避免比值列的极端值主导）
    sc = StandardScaler().fit(X[FRAC_COLS].to_numpy(float))
    ref = sc.transform(X[FRAC_COLS].to_numpy(float))
    dd = np.sqrt(((ref[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dd, np.inf)
    thr = float(np.percentile(dd.min(axis=1), 95))
    qry = sc.transform(X_ds[FRAC_COLS].to_numpy(float))
    d = np.sqrt(((qry[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    ds["hcj_kA_m_pred"] = np.round(pred_ds, 1)
    ds["hcj_pred_dex"] = np.round(to_log(pred_ds), 3)
    ds["ad_dist"] = np.round(d, 3)
    ds["in_domain"] = d <= thr
    ds["hcj_label_in_train"] = ds["in_train"]
    ds["hard_family"] = ds["space_group_symbol"].map(
        lambda s: SG_FAMILY.get(str(s), "other"))
    ds["hcj_trustworthy"] = ds["in_domain"] & ds["hard_family"].ne("other")

    out = ds[["material_id", "formula_pretty", "chemsys", "ordering", "Ms_T",
              "is_stable", "energy_above_hull_eV", "raw_cost_index_usd_per_kg",
              "hcj_kA_m_pred", "hcj_pred_dex", "ad_dist", "in_domain",
              "hard_family", "hcj_trustworthy", "hcj_label_in_train"]].copy()
    out = out.sort_values("hcj_kA_m_pred", ascending=False).reset_index(drop=True)
    out.to_csv(os.path.join(DATA_DIR, "27_hcj_imputed_design_space.csv"),
               index=False, encoding="utf-8-sig")
    n_dom = int(out["in_domain"].sum())
    print(f"WROTE 27_hcj_imputed_design_space.csv  104 行，"
          f"其中 {n_dom} 行落在适用域内（到训练集最近邻距离 <= {thr:.3f}，"
          "该阈值取训练集自身留一最近邻距离的 95 分位）")
    n_trust = int(out["hcj_trustworthy"].sum())
    print(f"进一步要求落在已知硬磁家族（2:14:1 / 2:17 / 1:12 / 1:5）后，"
          f"可用的行数为 {n_trust}")
    print("家族构成：", dict(out["hard_family"].value_counts()))
    print(f"注：设计空间里本身有实验 Hcj 的组分 {int(out['hcj_label_in_train'].sum())} 行")
    print(out.head(15).round(2).to_string(index=False))

    line("7. 保存指标并登记")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(DATA_DIR, "26_hcj_model_metrics.csv"),
                      index=False, encoding="utf-8-sig")
    print(metrics_df.round(3).to_string(index=False))

    # ---- 诊断图：留一预测 vs 实验 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib 不可用，跳过绘图：{exc}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
        ax = axes[0]
        ok_pts = ~agg["is_refeb"].to_numpy()
        ax.scatter(z[ok_pts], pg[ok_pts], s=14, c="#b0b7c3", alpha=0.6,
                   label=f"other RE compounds (n={int(ok_pts.sum())})")
        m = agg["is_refeb"].to_numpy()
        ax.scatter(z[m], pg[m], s=26, c="#c0392b", alpha=0.85,
                   label=f"RE-Fe-B (n={int(m.sum())})")
        lim = [min(z.min(), pg.min()) - 0.2, max(z.max(), pg.max()) + 0.2]
        ax.plot(lim, lim, ls="--", lw=1, c="#555555")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(r"experimental $\log_{10}(1+H_{cj}\,[\mathrm{kA/m}])$")
        ax.set_ylabel(r"predicted (grouped 5-fold)")
        ax.set_title("Hcj: composition-only model")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.25)
        ax = axes[1]
        resid = pg - z
        ax.scatter(X["frac_Fe"], resid, s=14, c="#3b6ea5", alpha=0.55)
        ax.axhline(0, ls="--", lw=1, c="#555555")
        ax.set_xlabel("Fe atomic fraction")
        ax.set_ylabel("residual (dex)")
        ax.set_title("Residual vs Fe fraction")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        png = os.path.join(DATA_DIR, "26_hcj_parity.png")
        fig.savefig(png)
        plt.close(fig)
        print(f"WROTE {png}")

    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keys = ["26_hcj_model_metrics.csv", "27_hcj_imputed_design_space.csv"]
        keep = idx[~idx["file"].isin(keys)]
        new = pd.DataFrame([
            {"file": "26_hcj_model_metrics.csv", "layer": "建模",
             "rows": len(metrics_df), "columns": len(metrics_df.columns),
             "source": "外部 Hcj 数据（scripts/train_hcj.py）",
             "purpose": "矫顽力模型交叉验证、子群误差与敏感性对照指标",
             "refresh_note": "重跑 scripts/train_hcj.py 刷新"},
            {"file": "27_hcj_imputed_design_space.csv", "layer": "标签",
             "rows": len(out), "columns": len(out.columns),
             "source": "18 + 模型补全（scripts/train_hcj.py）",
             "purpose": "设计空间每行的预测矫顽力、适用域距离与域内标记",
             "refresh_note": "重跑 scripts/train_hcj.py 刷新"},
        ])
        pd.concat([keep, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print("登记 2 行 -> 00_dataset_index.csv")

    dict_path = os.path.join(DATA_DIR, "00_data_dictionary.csv")
    if os.path.exists(dict_path):
        dic = pd.read_csv(dict_path, encoding="utf-8-sig", dtype=str).fillna("")
        dic = dic[~dic["file"].isin(["26_hcj_model_metrics.csv",
                                     "27_hcj_imputed_design_space.csv"])]
        add = pd.DataFrame([
            {"file": "26_hcj_model_metrics.csv", "column": "r2_log",
             "meaning": "对数尺度决定系数（目标为 log10(1+Hcj)）", "unit_or_range": "-inf~1"},
            {"file": "26_hcj_model_metrics.csv", "column": "mae_log_dex",
             "meaning": "对数尺度平均绝对误差", "unit_or_range": "dex（10 倍为一个 dex）"},
            {"file": "26_hcj_model_metrics.csv", "column": "mae_raw_kA_m",
             "meaning": "还原到原始尺度的平均绝对误差", "unit_or_range": "kA/m"},
            {"file": "26_hcj_model_metrics.csv", "column": "median_abs_err_ratio",
             "meaning": "中位相对误差 |pred-obs|/obs", "unit_or_range": "0~1"},
            {"file": "26_hcj_model_metrics.csv", "column": "cv",
             "meaning": "验证口径：kfold=随机 5 折；chemsys=按化学体系分组 5 折；"
                        "chemsys::RE-Fe-B=分组折上 RE-Fe-B 子群误差", "unit_or_range": "-"},
            {"file": "27_hcj_imputed_design_space.csv", "column": "hcj_kA_m_pred",
             "meaning": "模型预测的内禀矫顽力", "unit_or_range": "kA/m"},
            {"file": "27_hcj_imputed_design_space.csv", "column": "hcj_pred_dex",
             "meaning": "预测值取 log10(1+Hcj) 后的值", "unit_or_range": "dex"},
            {"file": "27_hcj_imputed_design_space.csv", "column": "ad_dist",
             "meaning": "到训练集最近邻的标准化成分距离（适用域判据）", "unit_or_range": ">=0"},
            {"file": "27_hcj_imputed_design_space.csv", "column": "in_domain",
             "meaning": "是否落在适用域内（ad_dist<=训练集留一最近邻距离 95 分位）",
             "unit_or_range": "0/1"},
            {"file": "27_hcj_imputed_design_space.csv", "column": "hard_family",
             "meaning": "由空间群推得的已知硬磁家族", "unit_or_range": "2:14:1 / 2:17 / 1:12 / 1:5 / other"},
            {"file": "27_hcj_imputed_design_space.csv", "column": "hcj_trustworthy",
             "meaning": "可信标记：in_domain 且属已知硬磁家族（推荐只使用这部分）",
             "unit_or_range": "0/1"},
            {"file": "27_hcj_imputed_design_space.csv", "column": "hcj_label_in_train",
             "meaning": "该组分本身出现在 Hcj 训练集里（其预测为样本内，不可当检验用）",
             "unit_or_range": "0/1"},
        ])
        pd.concat([dic, add], ignore_index=True).to_csv(
            dict_path, index=False, encoding="utf-8-sig")
        print(f"登记 {len(add)} 行 -> 00_data_dictionary.csv")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
