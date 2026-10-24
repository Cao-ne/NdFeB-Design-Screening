# 本脚本提供本地网页服务，并把化学式预测请求转交给已有的 Ms、Tc、Hcj 模型。
"""运行：python Demo/app.py"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from pymatgen.core import Composition

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
WEB_DIR = DEMO_DIR / "web"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Demo.common.data import add_features  # noqa: E402
from Demo.common.model import predict_package  # noqa: E402
from Demo.common.neighbors import get_indices  # noqa: E402
from Demo.Hcj.bound import attach_bound  # noqa: E402


MODEL_SPECS = (
    {
        "key": "Ms",
        "name": "饱和磁化强度",
        "unit": "emu/g",
        "note": "反映磁体理论上可达到的磁化强度上限。",
    },
    {
        "key": "Tc",
        "name": "居里温度",
        "unit": "K",
        "note": "高于该温度后，材料的有序磁性会消失。",
    },
    {
        "key": "Hcj",
        "name": "内禀矫顽力",
        "unit": "kA/m",
        "note": "受晶界和微结构影响显著，当前仅用于成分粗筛。",
    },
)

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/assets")
app.json.ensure_ascii = False


def load_artifact(spec: dict) -> dict:
    """读取单个模型及其训练元数据，缺失时立即给出明确错误。"""
    output_dir = DEMO_DIR / spec["key"] / "output"
    model_path = output_dir / "model.joblib"
    meta_path = output_dir / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"{spec['key']} 模型产物缺失，请先运行 python Demo/{spec['key']}/train.py"
        )

    import json

    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "package": joblib.load(model_path),
        "metadata": metadata,
    }


ARTIFACTS = {spec["key"]: load_artifact(spec) for spec in MODEL_SPECS}
NEIGHBORS = get_indices()


def _finite_or_none(value) -> float | None:
    """把 NaN/inf 转成 None，保证 JSON 可序列化。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@app.get("/")
def index():
    """返回深色单页。"""
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/health")
def health():
    """供启动检查确认三个模型均已加载。"""
    return jsonify(
        {
            "ok": True,
            "models": {
                spec["key"]: {
                    "loaded": True,
                    "rows_used": ARTIFACTS[spec["key"]]["metadata"].get("rows_used"),
                }
                for spec in MODEL_SPECS
            },
        }
    )


@app.post("/api/predict")
def predict():
    """校验化学式，返回三项预测、80% 区间和适用域状态。"""
    payload = request.get_json(silent=True) or {}
    raw_formula = str(payload.get("formula", "")).strip()
    formula = re.sub(r"\s+", "", raw_formula)

    if not formula:
        return jsonify({"error": "请输入化学式。"}), 400
    if len(formula) > 80:
        return jsonify({"error": "化学式过长，请检查输入。"}), 400

    try:
        composition = Composition(formula)
        if not composition.elements:
            raise ValueError("化学式不含有效元素")
    except Exception:  # noqa: BLE001
        return jsonify({"error": "化学式无法解析，请检查元素符号与数字格式。"}), 400

    try:
        features = add_features(pd.DataFrame({"formula": [formula]}))
        results = []
        for spec in MODEL_SPECS:
            artifact = ARTIFACTS[spec["key"]]
            prediction_frame = predict_package(artifact["package"], features)
            prediction = prediction_frame.iloc[0]
            values = {
                "value": float(prediction["pred"]),
                "lower_80": float(prediction["lower_80"]),
                "upper_80": float(prediction["upper_80"]),
                "domain_distance": float(prediction["domain_distance"]),
            }
            if spec["key"] == "Hcj":
                bound = attach_bound(prediction_frame.copy(), [formula]).iloc[0]
                values["pred_bounded"] = float(bound["pred_bounded"])
                values["bound_hit"] = bool(bound["bound_hit"])
                values["ha_kA_m"] = _finite_or_none(bound["ha_kA_m"])
                values["me"] = _finite_or_none(bound["me"])
                values["me_status"] = str(bound["me_status"])
            else:
                values["pred_bounded"] = values["value"]
                values["bound_hit"] = False
                values["ha_kA_m"] = None
                values["me"] = None
                values["me_status"] = None

            numeric = [
                values[field]
                for field in ("value", "lower_80", "upper_80", "domain_distance", "pred_bounded", "ha_kA_m", "me")
            ]
            if not all(value is None or math.isfinite(value) for value in numeric):
                raise ValueError(f"{spec['key']} 返回了非有限数值")

            metadata = artifact["metadata"]
            results.append(
                {
                    "key": spec["key"],
                    "name": spec["name"],
                    "unit": spec["unit"],
                    "note": spec["note"],
                    **values,
                    "in_domain": bool(prediction["in_domain"]),
                    "nearest": NEIGHBORS[spec["key"]].query(formula),
                    "model_created_at": metadata.get("created_at"),
                    "data_version": str(metadata.get("source_sha256", ""))[:12],
                }
            )
    except Exception as error:  # noqa: BLE001
        app.logger.exception("预测失败")
        return jsonify({"error": f"模型预测失败：{error}"}), 500

    return jsonify(
        {
            "formula": formula,
            "results": results,
            "all_in_domain": all(item["in_domain"] for item in results),
            "disclaimer": "以上均为模型预测值，不是实验实测结果。",
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", "5000"))
    print(f"本地 Demo 已启动：http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
