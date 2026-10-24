# 本文件负责读取和过滤 Tc 数据，并加载公共成分特征。
"""Tc 数据入口。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Demo.common.data import add_features, add_log, normalize_table, read_raw

SOURCE = "dataset/NEMAD/processed/tc_dataset.csv"

FLAGGED = {
    ("Nd2Fe14B", 450.0),
    ("Nd2Fe14B", 768.15),
    ("Nd2Fe17", 743.0),
    ("Ce2Fe14B", 490.0),
    ("Er2Fe14B", 314.0),
    ("Y2Co17", 940.0),
}

TASK = {
    "name": "Tc",
    "task_cn": "居里温度",
    "unit": "K",
    "transform": "none",
    "cv_splits": 5,
    "early_stopping_rounds": 60,
    "params": {
        "objective": "reg:pseudohubererror",
        "huber_slope": 1.0,
        "eval_metric": "mae",
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
        "n_estimators": 600,
    },
}


def load_data() -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """返回清洗后的 Tc 表、处理日志和重复标签冲突表。"""
    raw = read_raw(SOURCE)
    logs: list[dict] = []
    before = len(raw)
    raw["tc_K"] = pd.to_numeric(raw["tc_K"], errors="coerce")

    flagged_mask = raw.apply(
        lambda row: (str(row["formula"]), float(row["tc_K"])) in FLAGGED
        if pd.notna(row["tc_K"])
        else False,
        axis=1,
    )
    add_log(
        logs,
        "remove_known_conflicts",
        before,
        before - int(flagged_mask.sum()),
        "删除已确认的相同化学式温度冲突记录",
    )

    work = raw[~flagged_mask].copy()
    work = work[
        work["tc_K"].notna()
        & np.isfinite(work["tc_K"])
        & work["tc_K"].between(10, 2000)
    ]
    add_log(
        logs,
        "quality_filter",
        before - int(flagged_mask.sum()),
        len(work),
        "只保留 10-2000 K 的有效 Tc",
    )

    table, conflicts = normalize_table(
        work, formula_col="formula", value_col="tc_K", logs=logs
    )
    return add_features(table), logs, conflicts
