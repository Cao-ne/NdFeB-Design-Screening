# 本文件负责 XGBoost 分组交叉验证、预测区间、适用域、模型保存和单配方预测。
"""三个物理量共用的训练和预测工具。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Demo.common.data import FEATURE_COLS

SEED = 42


def transform_target(values, transform: str) -> np.ndarray:
    """把目标变换到适合建模的尺度。"""
    array = np.asarray(values, dtype=float)
    if transform == "log10_1p":
        if np.any(array < 0):
            raise ValueError("log10(1+Hcj) 要求 Hcj 不小于 0")
        return np.log10(1.0 + array)
    if transform != "none":
        raise ValueError(f"未知目标变换：{transform}")
    return array


def inverse_target(values, transform: str) -> np.ndarray:
    """把模型输出还原为页面显示单位。"""
    array = np.asarray(values, dtype=float)
    if transform == "log10_1p":
        return np.clip(np.power(10.0, array) - 1.0, 0.0, None)
    if transform != "none":
        raise ValueError(f"未知目标变换：{transform}")
    return array


def make_model(
    params: dict,
    n_estimators: int | None = None,
    early_stopping_rounds: int | None = None,
) -> xgb.XGBRegressor:
    """按任务参数创建 XGBoost；树模型不使用激活函数和支持向量。"""
    model_params = dict(params)
    model_params["random_state"] = SEED
    model_params["n_jobs"] = 4
    model_params["tree_method"] = "hist"
    if n_estimators is not None:
        model_params["n_estimators"] = int(n_estimators)
    if early_stopping_rounds is not None:
        model_params["early_stopping_rounds"] = int(early_stopping_rounds)
    return xgb.XGBRegressor(**model_params)


def task_metrics(task: str, y_true, y_pred) -> dict:
    """计算任务指标；Hcj 同时记录对数尺度和原始尺度误差。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }
    if task == "Hcj":
        true_log = np.log10(1.0 + np.clip(y_true, 0.0, None))
        pred_log = np.log10(1.0 + np.clip(y_pred, 0.0, None))
        positive = true_log > 1e-12
        result["mae_log_dex"] = float(mean_absolute_error(true_log, pred_log))
        result["median_abs_log_error"] = (
            float(np.median(np.abs(pred_log[positive] - true_log[positive])))
            if positive.any()
            else np.nan
        )
    return result


def fit_fold(
    table: pd.DataFrame,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    task_config: dict,
) -> tuple[xgb.XGBRegressor, np.ndarray, np.ndarray, dict]:
    """在一折上训练并返回验证集原始尺度预测。"""
    x_train = table.loc[train_index, FEATURE_COLS]
    x_valid = table.loc[valid_index, FEATURE_COLS]
    y_train_raw = table.loc[train_index, "y"].to_numpy(dtype=float)
    y_valid_raw = table.loc[valid_index, "y"].to_numpy(dtype=float)

    y_train = transform_target(y_train_raw, task_config["transform"])
    y_valid = transform_target(y_valid_raw, task_config["transform"])
    target_mean = float(np.mean(y_train))
    target_std = float(np.std(y_train))
    if target_std <= 1e-12:
        target_std = 1.0

    model = make_model(
        task_config["params"],
        early_stopping_rounds=task_config.get("early_stopping_rounds", 80),
    )
    model.fit(
        x_train,
        (y_train - target_mean) / target_std,
        eval_set=[(x_valid, (y_valid - target_mean) / target_std)],
        verbose=False,
    )
    pred_scaled = model.predict(x_valid)
    pred_transformed = pred_scaled * target_std + target_mean
    pred_raw = inverse_target(pred_transformed, task_config["transform"])
    fold_info = {
        "n_train": int(len(train_index)),
        "n_valid": int(len(valid_index)),
        "best_iteration": int(getattr(model, "best_iteration", 0)),
        "target_mean": target_mean,
        "target_std": target_std,
    }
    return model, pred_raw, pred_transformed, fold_info


def fit_domain_model(features: pd.DataFrame) -> dict:
    """建立标准化成分空间和 95% 最近邻距离阈值。"""
    domain_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    scaled = domain_pipe.fit_transform(features[FEATURE_COLS])
    n_neighbors = 2 if len(scaled) > 1 else 1
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(scaled)
    distances, _ = neighbors.kneighbors(scaled)
    reference_distance = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
    return {
        "pipe": domain_pipe,
        "neighbors": neighbors,
        "limit": float(np.quantile(reference_distance, 0.95)),
    }


def save_json(path: Path, payload: dict) -> None:
    """以 UTF-8 保存可读 JSON。"""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def train_and_save(
    table: pd.DataFrame,
    logs: list[dict],
    conflicts: pd.DataFrame,
    task_config: dict,
    source_path: Path,
    source_hash: str,
    output_dir: Path,
) -> dict:
    """完成分组交叉验证、最终训练、不确定性校准并保存全部产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_groups = table["chem_sys"].nunique()
    n_splits = min(int(task_config.get("cv_splits", 5)), n_groups)
    if n_splits < 2:
        raise ValueError(f"{task_config['name']} 的化学体系不足 2 组，无法分组验证")

    splitter = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(table), np.nan, dtype=float)
    oof_pred_transformed = np.full(len(table), np.nan, dtype=float)
    fold_rows = []
    best_iterations = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(table, groups=table["chem_sys"])
    ):
        model, pred_raw, pred_transformed, fold_info = fit_fold(
            table, np.asarray(train_index), np.asarray(valid_index), task_config
        )
        oof_pred[valid_index] = pred_raw
        oof_pred_transformed[valid_index] = pred_transformed
        best_iterations.append(fold_info["best_iteration"])

        valid = table.iloc[valid_index]
        metrics = task_metrics(task_config["name"], valid["y"], pred_raw)
        fold_rows.append({"fold": fold, **fold_info, **metrics})
        print(
            f"  {task_config['name']} fold {fold}: "
            f"MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}"
        )

    y_true = table["y"].to_numpy(dtype=float)
    y_transformed = transform_target(y_true, task_config["transform"])
    residual_transformed = np.abs(y_transformed - oof_pred_transformed)
    interval_80 = float(np.quantile(residual_transformed, 0.80))
    coverage = float(np.mean(residual_transformed <= interval_80))

    domain = fit_domain_model(table)
    final_iters = max(1, int(np.median(best_iterations)) + 1)
    final_target_mean = float(np.mean(y_transformed))
    final_target_std = float(np.std(y_transformed))
    if final_target_std <= 1e-12:
        final_target_std = 1.0

    final_model = make_model(task_config["params"], n_estimators=final_iters)
    final_model.fit(
        table[FEATURE_COLS],
        (y_transformed - final_target_mean) / final_target_std,
        verbose=False,
    )

    package = {
        "task": task_config["name"],
        "task_cn": task_config["task_cn"],
        "unit": task_config["unit"],
        "transform": task_config["transform"],
        "feature_cols": FEATURE_COLS,
        "target_mean": final_target_mean,
        "target_std": final_target_std,
        "interval_80_transformed": interval_80,
        "domain": domain,
        "model": final_model,
        "n_estimators": final_iters,
    }
    joblib.dump(package, output_dir / "model.joblib")

    oof = table[["formula", "chem_sys", "y", "n_records", "y_min", "y_max"]].copy()
    oof["pred"] = oof_pred
    oof["pred_transformed"] = oof_pred_transformed
    oof["abs_error"] = np.abs(oof["y"] - oof["pred"])
    oof["abs_error_transformed"] = residual_transformed
    oof["inside_interval_80"] = residual_transformed <= interval_80
    oof.to_csv(output_dir / "oof.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(fold_rows).to_csv(
        output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(logs).to_csv(
        output_dir / "data_log.csv", index=False, encoding="utf-8-sig"
    )
    conflicts.to_csv(output_dir / "conflicts.csv", index=False, encoding="utf-8-sig")

    importance = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(
        output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    overall = task_metrics(task_config["name"], y_true, oof_pred)
    metadata = {
        "task": task_config["name"],
        "task_cn": task_config["task_cn"],
        "unit": task_config["unit"],
        "target_transform": task_config["transform"],
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows_used": int(len(table)),
        "chemical_systems": int(n_groups),
        "cv_splits": int(n_splits),
        "overall_metrics": overall,
        "coverage80": coverage,
        "domain_distance_limit": domain["limit"],
        "final_n_estimators": final_iters,
        "params": task_config["params"],
        "activation": "不适用：XGBoost 为树模型",
        "support_vectors": "不适用：未使用 SVR",
        "features": FEATURE_COLS,
    }
    save_json(output_dir / "meta.json", metadata)
    print(
        f"{task_config['name']} 完成：{len(table)} 行，"
        f"总体 MAE={overall['mae']:.4f}，R2={overall['r2']:.4f}，"
        f"80% 区间覆盖率={coverage:.3f}"
    )
    return metadata


def predict_package(package: dict, features: pd.DataFrame) -> pd.DataFrame:
    """使用已保存模型生成点预测、80% 区间和适用域标记。"""
    expected = package["feature_cols"]
    matrix = features.reindex(columns=expected)
    scaled = package["model"].predict(matrix)
    pred_transformed = scaled * package["target_std"] + package["target_mean"]
    half_width = package["interval_80_transformed"]
    low_transformed = pred_transformed - half_width
    high_transformed = pred_transformed + half_width

    transform = package["transform"]
    pred = inverse_target(pred_transformed, transform)
    low = inverse_target(low_transformed, transform)
    high = inverse_target(high_transformed, transform)

    domain_pipe = package["domain"]["pipe"]
    query_scaled = domain_pipe.transform(matrix)
    distance, _ = package["domain"]["neighbors"].kneighbors(query_scaled, n_neighbors=1)
    distance = distance[:, 0]

    return pd.DataFrame(
        {
            "pred": pred,
            "lower_80": low,
            "upper_80": high,
            "domain_distance": distance,
            "in_domain": distance <= package["domain"]["limit"],
        }
    )
