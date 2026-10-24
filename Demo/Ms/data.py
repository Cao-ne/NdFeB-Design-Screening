# 本文件负责读取和过滤 Ms 数据，并加载公共成分特征。
"""Ms 数据入口。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Demo.common.data import add_features, add_log, normalize_table, read_raw

SOURCE = "dataset/NEMAD/processed/ms_cleaned.csv"

TASK = {
    "name": "Ms",
    "task_cn": "饱和磁化强度",
    "unit": "emu/g",
    "transform": "none",
    "cv_splits": 5,
    "early_stopping_rounds": 80,
    "params": {
        "objective": "reg:pseudohubererror",
        "huber_slope": 1.0,
        "eval_metric": "mae",
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "n_estimators": 800,
    },
}


def load_data() -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """返回清洗后的 Ms 表、处理日志和重复标签冲突表。"""
    raw = read_raw(SOURCE)
    logs: list[dict] = []
    before = len(raw)
    raw["Ms_emu_g"] = pd.to_numeric(raw["Ms_emu_g"], errors="coerce")
    filtered = raw[
        raw["Ms_conv_status"].eq("ok")
        & raw["Ms_emu_g"].notna()
        & np.isfinite(raw["Ms_emu_g"])
        & raw["Ms_emu_g"].ge(0)
    ].copy()
    add_log(
        logs,
        "quality_filter",
        before,
        len(filtered),
        "只保留单位换算状态为 ok 且 Ms 非负的记录",
    )
    table, conflicts = normalize_table(
        filtered, formula_col="material_name", value_col="Ms_emu_g", logs=logs
    )
    return add_features(table), logs, conflicts
