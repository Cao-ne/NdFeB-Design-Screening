# 本脚本把 NEMAD 高矫顽力档与文献抽取记录整理成各向异性场 Ha 数据集（逐条表、聚合表、日志、冲突表）。
"""构建 Ha（各向异性场）数据集 v0。

输入（只读，不修改）：
  dataset/NEMAD/processed/hcj_cleaned.csv
      hcj_value_class == likely_anisotropy_field 的记录（>60 kOe）
      按上游 QC 的判定，这一档实际是各向异性场 Ha，而不是矫顽力。
  dataset/shared/raw/03_literature_records_raw.csv
      property == anisotropy_field_Ha 的记录（文献直接报告的 Ha）。

输出：
  dataset/shared/processed/33_ha_records.csv    逐条记录（含出处、页面、证据句）
  dataset/shared/processed/33_ha_dataset.csv    按配方聚合（中位数、极差、温度档）
  dataset/shared/processed/33_ha_data_log.csv   每一步行数变化
  dataset/shared/processed/33_ha_conflicts.csv  同配方标签分歧

单位约定：Ha 主单位 kA/m，附 T；1 T = 1/μ0 = 795.7747 kA/m。
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "Papers"
PDFTOTEXT = "pdftotext"

# 从文献正文/表格里按正则可核验地抽取 K1 与 Js，再换算成 Ha
# 每条规则必须同时命中 k1 与 js，命中失败即报错，避免静默漏抽
LIT_RULES = [
    {
        "pdf": "PNAS_ML引导发现无稀土Fe3CoB2磁体.pdf",
        "composition": "Fe3CoB2",
        "k1_unit": "MJ/m3",
        "k1_pattern": r"K1 = (1\.0) MJ/m3",
        "js_pattern": r"Js = (1\.35) T",
        "temperature_K": 300.0,
        "data_type": "measured",
        "note": "正文给出 Ha=2K1/Ms=1.86 T，与本地换算一致",
    },
    {
        "pdf": "PNAS_ML引导发现无稀土Fe3CoB2磁体.pdf",
        "composition": "Fe3CoB2",
        "k1_unit": "MJ/m3",
        "k1_pattern": r"K1 = (1\.2) MJ/m3 and Js = 1\.39 T",
        "js_pattern": r"Js = (1\.39) T",
        "temperature_K": 10.0,
        "data_type": "measured",
        "note": "低温测量值",
    },
    {
        "pdf": "2101.10773_数据驱动发现Co3Mn2Ge无稀土永磁.pdf",
        "composition": "Co3Mn2Ge",
        "k1_unit": "MJ/m3",
        "k1_pattern": r"anisotropy constant of (1\.18) MJ/m3",
        "js_pattern": r"polarization of (0\.86) T at 10 K",
        "temperature_K": 10.0,
        "data_type": "measured",
        "note": "10 K 实验值",
    },
    {
        "pdf": "1903.11995_少稀土永磁体计算设计.pdf",
        "composition": "Fe3Sn0.75Sb0.25",
        "k1_unit": "MJ/m3",
        "k1_pattern": r"anisotropy constant\s+K = (0\.33) MJ/m3",
        "js_pattern": r"spontaneous magnetization\s+[µμ]0Ms = (1\.52) T",
        "temperature_K": None,
        "data_type": "computed",
        "note": "RSPt 计算值，温度未在句中标明",
    },
    {
        "pdf": "2312.02475_ML预测永磁体矫顽力.pdf",
        "composition": "Nd2Fe14B",
        "k1_unit": "MJ/m3",
        "k1_pattern": r"Ku\s+(6\.16)\s+1\.21",
        "js_pattern": r"Ms\s+(1\.31)\s+1\.24",
        "js_unit": "MA/m",
        "temperature_K": None,
        "data_type": "computed",
        "note": "TABLE VIII，PBE+U+SOC 计算；Ms 单位 MA/m",
    },
    {
        "pdf": "2312.02475_ML预测永磁体矫顽力.pdf",
        "composition": "Ce2Fe14B",
        "k1_unit": "MJ/m3",
        "k1_pattern": r"Ku\s+6\.16\s+(1\.21)",
        "js_pattern": r"Ms\s+1\.31\s+(1\.24)",
        "js_unit": "MA/m",
        "temperature_K": None,
        "data_type": "computed",
        "note": "TABLE VIII，PBE+U+SOC 计算；Ms 单位 MA/m",
    },
    {
        "pdf": "PRMater_无稀土磁性材料数据库_Ames实验室.pdf",
        "composition": "Co3N",
        "k1_unit": "MJ/m3",
        "k1_pattern": r"K1 = (1\.04) MJ/m3",
        "js_pattern": r"Js = (0\.73) T",
        "temperature_K": None,
        "data_type": "measured",
        "note": "论文报告的实验值（Tc 约 450 K）",
    },
    {
        "pdf": "PRMater_无稀土磁性材料数据库_Ames实验室.pdf",
        "composition": "Zr2Co11",
        "k1_unit": "Mergs/cm3",
        "k1_pattern": r"(13\.5) Mergs/cm3",
        "js_pattern": r"Js = (9\.7) kG",
        "js_unit": "kG",
        "temperature_K": None,
        "data_type": "measured",
        "note": "实验值，原始单位 Merg/cm3 与 kG",
    },
    # 以下四条为单晶实验直接报告的 HA，不需要由 K1 换算
    {
        "pdf": "1508.07792_Ce取代Nd2Fe14B单晶生长与表征.pdf",
        "composition": "Nd2Fe14B",
        "ha_pattern": r"At 300 K the\s+value of HA drops from an undoped value of (8\.1) T",
        "temperature_K": 300.0,
        "data_type": "measured",
        "note": "单晶硬轴磁化曲线外推得到的 HA；300 K 报 8.1 T，高于常用的 6.7 T，口径不同",
    },
    {
        "pdf": "1508.07792_Ce取代Nd2Fe14B单晶生长与表征.pdf",
        "composition": "(Nd0.62Ce0.38)2Fe14B",
        "ha_pattern": r"value of (6\.2) T for the highest achievable Ce",
        "temperature_K": 300.0,
        "data_type": "measured",
        "note": "Ce 取代量最高可及成分，300 K",
    },
    {
        "pdf": "1508.07792_Ce取代Nd2Fe14B单晶生长与表征.pdf",
        "composition": "Nd2Fe14B",
        "ha_pattern": r"At 400 K HA drops from (5\.5) T",
        "temperature_K": 400.0,
        "data_type": "measured",
        "note": "400 K，属高温档",
    },
    {
        "pdf": "1508.07792_Ce取代Nd2Fe14B单晶生长与表征.pdf",
        "composition": "(Nd0.62Ce0.38)2Fe14B",
        "ha_pattern": r"to (4\.7) T for crystals",
        "temperature_K": 400.0,
        "data_type": "measured",
        "note": "400 K，属高温档",
    },
]

K1_TO_MJ = {"MJ/m3": 1.0, "Mergs/cm3": 0.1}
JS_TO_T = {"T": 1.0, "kG": 0.1, "MA/m": 1.2566370614}

MU0 = 4e-7 * math.pi
KA_PER_T = (1.0 / MU0) / 1000.0  # 795.7747 kA/m per T
KA_PER_OE = 0.0795775

NEMAD_HCJ = ROOT / "dataset/NEMAD/processed/hcj_cleaned.csv"
LIT_RECORDS = ROOT / "dataset/shared/raw/03_literature_records_raw.csv"
OUT_DIR = ROOT / "dataset/shared/processed"

# Ha 合理性区间（kA/m）：0.13 T ~ 75 T，超出记为待复核
HA_MIN_KA_M = 100.0
HA_MAX_KA_M = 60000.0
HA_OUTLIER_KA_M = 30000.0  # >37.7 T，硬磁 Ha 的罕见上界

# 已核实的抽取问题：命中后不参与聚合，只在逐条表里保留以便追溯
KNOWN_EXTRACTION_ISSUES = {
    ("PNAS_ML引导发现无稀土Fe3CoB2磁体.pdf", 1.35): (
        "数值与论文给出的 Js=1.35 T 相同，疑为把 Js 抽成 Ha；"
        "论文按 Ha=2K1/Ms 给出 1.86 T，与 1480 kA/m 一致"
    ),
}

# 已核实的手工备注：写入冲突表，避免把可疑值当干净标签
MANUAL_CONFLICTS = [
    {
        "formula": "Fe3CoB2",
        "n_records": 2,
        "ha_kA_m_median": 1277.22,
        "ha_T_median": 1.605,
        "ha_kA_m_min": 1074.30,
        "ha_kA_m_max": 1480.14,
        "max_over_min": 1.38,
        "n_rt": 0,
        "has_hts": False,
        "sources": "shared/raw/03_literature_records_raw.csv",
        "reason": "同源两条：1.35 T 与 Js=1.35 T 数值相同，疑为抽取串行；论文按 Ha=2K1/Ms 给出 1.86 T，与 1480 kA/m 一致，应以 1.86 T 为准",
    },
    {
        "formula": "ThMn12 系（Table II）",
        "n_records": 0,
        "ha_kA_m_median": None,
        "ha_T_median": None,
        "ha_kA_m_min": None,
        "ha_kA_m_max": None,
        "max_over_min": None,
        "n_rt": 0,
        "has_hts": False,
        "sources": "Papers/1803.03073_ML硬磁相成分优化_ThMn12.pdf Table II",
        "reason": "该表 Ha 栏按未缩放的 DFT K1 计算（如 2×169/1.45=233 T），物理上不可能；同表 K1 另有启发式缩放因子，故整表不纳入",
    },
]

DICT_ROWS = [
    ("shared/processed/33_ha_records.csv", "ha_id", "Ha 记录编号", "-"),
    ("shared/processed/33_ha_records.csv", "formula_raw", "上游原始材料名", "-"),
    ("shared/processed/33_ha_records.csv", "formula", "约化化学式", "-"),
    ("shared/processed/33_ha_records.csv", "ha_kA_m", "各向异性场", "kA/m"),
    ("shared/processed/33_ha_records.csv", "ha_T", "各向异性场", "T"),
    ("shared/processed/33_ha_records.csv", "ha_source", "来源类别", "nemad_class / literature"),
    ("shared/processed/33_ha_records.csv", "temperature_K", "测量温度", "K，缺失为空"),
    ("shared/processed/33_ha_records.csv", "temp_band", "温度档", "RT/low/high/NA"),
    ("shared/processed/33_ha_records.csv", "source_file", "来源文件", "-"),
    ("shared/processed/33_ha_records.csv", "source_doi", "来源 DOI 或 PDF 文件名", "-"),
    ("shared/processed/33_ha_records.csv", "evidence", "判定依据", "-"),
    ("shared/processed/33_ha_records.csv", "confidence", "置信度", "high/medium/low"),
    ("shared/processed/33_ha_records.csv", "needs_review", "是否需要人工复核原文献", "True/False"),
    ("shared/processed/33_ha_records.csv", "flag_outlier", "Ha 是否超过 30000 kA/m（约 37.7 T），只标记不删除", "True/False"),
    ("shared/processed/33_ha_records.csv", "flag_exclude", "是否因已核实的抽取错误而不参与聚合", "True/False"),
    ("shared/processed/33_ha_records.csv", "review_note", "复核备注", "-"),
    ("shared/processed/33_ha_records.csv", "data_type", "数据类型", "measured/computed/reported"),
    ("shared/processed/33_ha_records.csv", "page", "原文页码（仅文献表格抽取）", "整数或空"),
    ("shared/processed/33_ha_dataset.csv", "formula", "约化化学式", "-"),
    ("shared/processed/33_ha_dataset.csv", "n_records", "该配方 Ha 记录数", "整数"),
    ("shared/processed/33_ha_dataset.csv", "n_measured", "其中实测/文献上报条数", "整数"),
    ("shared/processed/33_ha_dataset.csv", "n_computed", "其中 DFT 计算条数", "整数"),
    ("shared/processed/33_ha_dataset.csv", "ha_kA_m_median", "Ha 中位数", "kA/m"),
    ("shared/processed/33_ha_dataset.csv", "ha_T_median", "Ha 中位数", "T"),
    ("shared/processed/33_ha_dataset.csv", "ha_kA_m_min", "Ha 最小值", "kA/m"),
    ("shared/processed/33_ha_dataset.csv", "ha_kA_m_max", "Ha 最大值", "kA/m"),
    ("shared/processed/33_ha_dataset.csv", "max_over_min", "极差比", ">=1"),
    ("shared/processed/33_ha_dataset.csv", "n_rt", "室温档记录数", "整数"),
    ("shared/processed/33_ha_dataset.csv", "median_basis", "中位数口径", "RT/measured/computed"),
    ("shared/processed/33_ha_dataset.csv", "has_hts", "化学式是否含 Dy/Tb", "True/False"),
    ("shared/processed/33_ha_dataset.csv", "sources", "来源文件集合", "-"),
    ("shared/processed/33_ha_data_log.csv", "step", "处理步骤", "-"),
    ("shared/processed/33_ha_data_log.csv", "rows_before", "步骤前行数", "整数"),
    ("shared/processed/33_ha_data_log.csv", "rows_after", "步骤后行数", "整数"),
    ("shared/processed/33_ha_data_log.csv", "removed", "删除行数", "整数"),
    ("shared/processed/33_ha_data_log.csv", "note", "步骤说明", "-"),
    ("shared/processed/33_ha_conflicts.csv", "formula", "冲突配方", "-"),
    ("shared/processed/33_ha_conflicts.csv", "n_records", "该配方记录数", "整数"),
    ("shared/processed/33_ha_conflicts.csv", "ha_kA_m_median", "Ha 中位数", "kA/m"),
    ("shared/processed/33_ha_conflicts.csv", "ha_T_median", "Ha 中位数", "T"),
    ("shared/processed/33_ha_conflicts.csv", "ha_kA_m_min", "Ha 最小值", "kA/m"),
    ("shared/processed/33_ha_conflicts.csv", "ha_kA_m_max", "Ha 最大值", "kA/m"),
    ("shared/processed/33_ha_conflicts.csv", "max_over_min", "同配方 Ha 极差比", ">=1"),
    ("shared/processed/33_ha_conflicts.csv", "n_rt", "室温档记录数", "整数"),
    ("shared/processed/33_ha_conflicts.csv", "has_hts", "是否含 Dy/Tb", "True/False"),
    ("shared/processed/33_ha_conflicts.csv", "sources", "来源文件集合", "-"),
    ("shared/processed/33_ha_conflicts.csv", "reason", "冲突或抽取问题的原因", "-"),
]


def reduced_formula(value: object) -> str | None:
    """用 pymatgen 约化化学式；不可用时退回正则，两种口径都记日志。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        from pymatgen.core import Composition

        return Composition(text).reduced_formula
    except Exception:  # noqa: BLE001
        return _fallback_reduced(text)


def _fallback_reduced(text: str) -> str | None:
    """无 pymatgen 时的最小实现，只处理不含括号的简单化学式。"""
    parts = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", text)
    if not parts:
        return None
    counts: dict[str, float] = {}
    for element, number in parts:
        counts[element] = counts.get(element, 0.0) + (float(number) if number else 1.0)
    values = [v for v in counts.values() if v > 0]
    if not values:
        return None
    gcd_value = values[0]
    for value in values[1:]:
        gcd_value = math.gcd(int(round(gcd_value * 1000)), int(round(value * 1000))) / 1000
    if gcd_value <= 0:
        return None
    out = []
    for element in sorted(counts):
        number = counts[element] / gcd_value
        if number <= 0:
            continue
        out.append(element + ("" if abs(number - 1) < 1e-9 else f"{number:g}"))
    return "".join(out)


def temp_band(value: float | None) -> str:
    """把测量温度压成三档，便于分组统计。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if 250.0 <= value <= 350.0:
        return "RT"
    if value < 250.0:
        return "low"
    return "high"


def to_kA_m(value: float, unit: str) -> float:
    """统一换算到 kA/m。"""
    unit_norm = str(unit).strip().lower().replace("μ0", "")
    if unit_norm in {"t", "tesla"}:
        return value * KA_PER_T
    if unit_norm in {"ka/m", "ka m-1", "kam-1"}:
        return value
    if unit_norm in {"ma/m"}:
        return value * 1000.0
    if unit_norm in {"oe", "kOe".lower()}:
        return value * KA_PER_OE
    raise ValueError(f"未识别的单位：{unit}")


def build_nemad_records(logs: list[dict]) -> pd.DataFrame:
    """读取 NEMAD 被判为各向异性场的记录，构造逐条表。"""
    raw = pd.read_csv(NEMAD_HCJ, low_memory=False)
    log(logs, "nemad_read", len(raw), len(raw), "读取 hcj_cleaned.csv")

    selected = raw[raw["hcj_value_class"].eq("likely_anisotropy_field")].copy()
    log(
        logs,
        "nemad_select_ha_class",
        len(raw),
        len(selected),
        "只保留 hcj_value_class=likely_anisotropy_field（>60 kOe，上游 QC 判为 Ha）",
    )

    selected["ha_kA_m"] = pd.to_numeric(selected["Hcj_kA_m"], errors="coerce")
    before = len(selected)
    selected = selected[selected["ha_kA_m"].notna() & (selected["ha_kA_m"] > 0)].copy()
    log(logs, "nemad_drop_missing_value", before, len(selected), "删除 Ha 缺失或非正记录")

    selected["formula"] = selected["material_name"].map(reduced_formula)
    before = len(selected)
    selected = selected[selected["formula"].notna()].copy()
    log(logs, "nemad_drop_unparsed_formula", before, len(selected), "删除无法解析化学式的记录")

    frame = pd.DataFrame(
        {
            "formula_raw": selected["material_name"].astype(str),
            "formula": selected["formula"],
            "ha_kA_m": selected["ha_kA_m"].astype(float),
            "ha_source": "nemad_class",
            "temperature_K": pd.to_numeric(selected.get("meas_temp_K"), errors="coerce"),
            "source_file": "NEMAD/processed/hcj_cleaned.csv",
            "source_doi": selected.get("source_doi", pd.Series([""] * len(selected))).astype(str),
            "evidence": (
                "NEMAD 将 "
                + selected["hcj_value_class"].astype(str)
                + " 判为各向异性场；原始 Hcj 栏 "
                + selected["hcj_kOe"].round(1).astype(str)
                + " kOe"
            ),
            "confidence": "medium",
            "needs_review": True,
            "flag_exclude": False,
            "review_note": "",
            "data_type": "measured",
        }
    )
    return frame


def build_literature_records(logs: list[dict]) -> pd.DataFrame:
    """读取文献抽取表中直接报告 Ha 的记录。"""
    raw = pd.read_csv(LIT_RECORDS, low_memory=False)
    selected = raw[raw["property"].eq("anisotropy_field_Ha")].copy()
    log(logs, "literature_select_ha", len(raw), len(selected), "只保留 property=anisotropy_field_Ha")

    rows = []
    for _, row in selected.iterrows():
        try:
            ha_kA_m = to_kA_m(float(row["value_SI"]), str(row["unit_SI"]))
        except Exception:  # noqa: BLE001
            continue
        composition = row.get("composition_nearest") or row.get("composition_raw")
        note = KNOWN_EXTRACTION_ISSUES.get((str(row.get("source_pdf", "")), float(row["value_SI"])))
        rows.append(
            {
                "formula_raw": str(composition),
                "formula": reduced_formula(composition),
                "ha_kA_m": ha_kA_m,
                "ha_source": "literature",
                "temperature_K": None,
                "source_file": "shared/raw/03_literature_records_raw.csv",
                "source_doi": str(row.get("source_pdf", "")),
                "evidence": str(row.get("evidence", ""))[:300],
                "confidence": "low" if note else str(row.get("confidence", "")),
                "needs_review": False,
                "flag_exclude": bool(note),
                "review_note": note or "",
            }
        )
    frame = pd.DataFrame(rows)
    log(logs, "literature_keep_parsed", len(selected), len(frame), "保留单位可换算、化学式可解析的记录")
    return frame


def pdf_pages(pdf_name: str) -> list[tuple[int, str]]:
    """把 PDF 转成按页的文本；返回 [(页码, 归一化文本)]。"""
    cache = Path(tempfile.gettempdir()) / "ha_pdf_text"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (Path(pdf_name).stem + ".txt")
    source = PAPERS / pdf_name
    if not target.exists():
        exe = PDFTOTEXT
        subprocess.run([exe, "-layout", str(source), str(target)], check=True)
    raw = target.read_text(encoding="utf-8", errors="ignore")
    pages = []
    for index, page in enumerate(raw.split("\f"), start=1):
        normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", page))
        pages.append((index, normalized))
    return pages


def build_literature_table_records(logs: list[dict]) -> pd.DataFrame:
    """按 LIT_RULES 从 PDF 正文与表格中抽取 K1/Js 并换算 Ha。"""
    rows = []
    for rule in LIT_RULES:
        pages = pdf_pages(rule["pdf"])
        if "ha_pattern" in rule:
            found = None
            found_page = None
            for number, text in pages:
                match = re.search(rule["ha_pattern"], text)
                if match and found is None:
                    found, found_page = match, number
            if found is None:
                raise ValueError(f"{rule['pdf']} 的 HA 抽取规则未命中：{rule.get('composition')}")
            ha_kA_m = float(found.group(1)) * KA_PER_T
            rows.append(
                {
                    "formula_raw": rule["composition"],
                    "formula": reduced_formula(rule["composition"]),
                    "ha_kA_m": ha_kA_m,
                    "ha_source": "literature_derived",
                    "temperature_K": rule["temperature_K"],
                    "source_file": f"Papers/{rule['pdf']}",
                    "source_doi": rule["pdf"],
                    "evidence": f"论文直接报告 HA={found.group(1)} T；换算 {ha_kA_m:.0f} kA/m",
                    "confidence": "high",
                    "needs_review": False,
                    "flag_exclude": False,
                    "review_note": rule["note"],
                    "data_type": rule["data_type"],
                    "page": found_page,
                }
            )
            continue

        k1_match = js_match = None
        k1_page = js_page = None
        for number, text in pages:
            found = re.search(rule["k1_pattern"], text)
            if found and k1_match is None:
                k1_match, k1_page = found, number
            found = re.search(rule["js_pattern"], text)
            if found and js_match is None:
                js_match, js_page = found, number
        if k1_match is None or js_match is None:
            raise ValueError(f"{rule['pdf']} 的抽取规则未命中：{rule.get('composition')}")

        k1 = float(k1_match.group(1)) * K1_TO_MJ[rule["k1_unit"]]
        js = float(js_match.group(1)) * JS_TO_T[rule.get("js_unit", "T")]
        ha_kA_m = 2000.0 * k1 / js
        rows.append(
            {
                "formula_raw": rule["composition"],
                "formula": reduced_formula(rule["composition"]),
                "ha_kA_m": ha_kA_m,
                "ha_source": "literature_derived",
                "temperature_K": rule["temperature_K"],
                "source_file": f"Papers/{rule['pdf']}",
                "source_doi": rule["pdf"],
                "evidence": f"K1={k1:g} MJ/m3, Js={js:g} T；Ha=2K1/Js={ha_kA_m:.0f} kA/m",
                "confidence": "medium" if rule["data_type"] == "computed" else "high",
                "needs_review": False,
                "flag_exclude": False,
                "review_note": rule["note"],
                "data_type": rule["data_type"],
                "page": k1_page,
            }
        )
    frame = pd.DataFrame(rows)
    log(logs, "literature_table_extract", len(LIT_RULES), len(frame), "按正则规则从 PDF 抽取 K1/Js")
    return frame


def log(logs: list[dict], step: str, before: int, after: int, note: str) -> None:
    """追加一条处理日志。"""
    logs.append(
        {
            "step": step,
            "rows_before": int(before),
            "rows_after": int(after),
            "removed": int(before - after),
            "note": note,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Ha 数据集")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="输出目录，默认 dataset/shared/processed")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs: list[dict] = []
    nemad = build_nemad_records(logs)
    literature = build_literature_records(logs)
    lit_table = build_literature_table_records(logs)
    records = pd.concat([nemad, literature, lit_table], ignore_index=True)
    records["data_type"] = records["data_type"].fillna("reported")
    log(
        logs,
        "merge_sources",
        len(nemad) + len(literature) + len(lit_table),
        len(records),
        "合并 NEMAD、文献直报与文献表格抽取三个来源",
    )

    records["ha_T"] = (records["ha_kA_m"] / KA_PER_T).round(4)
    records["ha_kA_m"] = records["ha_kA_m"].round(2)
    records["temp_band"] = records["temperature_K"].map(temp_band)
    records["ha_id"] = [f"HA-{i:04d}" for i in range(1, len(records) + 1)]

    before = len(records)
    records = records[(records["ha_kA_m"] >= HA_MIN_KA_M) & (records["ha_kA_m"] <= HA_MAX_KA_M)].copy()
    log(
        logs,
        "range_filter",
        before,
        len(records),
        f"只保留 {HA_MIN_KA_M:g}–{HA_MAX_KA_M:g} kA/m 的记录",
    )
    records["flag_outlier"] = records["ha_kA_m"] > HA_OUTLIER_KA_M
    log(
        logs,
        "outlier_flag",
        len(records),
        len(records),
        f">{HA_OUTLIER_KA_M:g} kA/m 的记录标 flag_outlier，不删除",
    )

    columns = [
        "ha_id",
        "formula_raw",
        "formula",
        "ha_kA_m",
        "ha_T",
        "ha_source",
        "temperature_K",
        "temp_band",
        "source_file",
        "source_doi",
        "evidence",
        "confidence",
        "needs_review",
        "flag_outlier",
        "flag_exclude",
        "review_note",
        "data_type",
        "page",
    ]
    records = records[columns].sort_values(["formula", "ha_kA_m"]).reset_index(drop=True)

    before = len(records)
    usable = records[~records["flag_exclude"]].copy()
    log(logs, "drop_known_extraction_issue", before, len(usable), "剔除已核实的抽取错误记录，逐条表保留原记录")

    grouped = (
        usable.groupby("formula", sort=True)
        .agg(
            n_records=("ha_kA_m", "size"),
            n_measured=("data_type", lambda s: int(s.ne("computed").sum())),
            n_computed=("data_type", lambda s: int(s.eq("computed").sum())),
            ha_kA_m_min=("ha_kA_m", "min"),
            ha_kA_m_max=("ha_kA_m", "max"),
            n_rt=("temp_band", lambda s: int((s == "RT").sum())),
            sources=("source_file", lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
    )
    # 中位数优先用室温实测；没有室温再用实测；都没有才用计算值
    rt_median = usable[usable["temp_band"].eq("RT")].groupby("formula")["ha_kA_m"].median()
    measured_median = (
        usable[usable["data_type"].ne("computed")].groupby("formula")["ha_kA_m"].median()
    )
    computed_median = (
        usable[usable["data_type"].eq("computed")].groupby("formula")["ha_kA_m"].median()
    )
    preferred = (
        rt_median.rename("ha_kA_m_median")
        .combine_first(measured_median.rename("ha_kA_m_median"))
        .combine_first(computed_median.rename("ha_kA_m_median"))
    )
    grouped = grouped.merge(preferred, left_on="formula", right_index=True, how="left")
    grouped["median_basis"] = [
        "RT" if f in rt_median.index else ("measured" if f in measured_median.index else "computed")
        for f in grouped["formula"]
    ]
    grouped["max_over_min"] = grouped["ha_kA_m_max"] / grouped["ha_kA_m_min"]
    grouped["ha_T_median"] = grouped["ha_kA_m_median"] / KA_PER_T
    grouped["has_hts"] = grouped["formula"].str.contains("Dy|Tb", regex=True)
    grouped = grouped[
        [
            "formula",
            "n_records",
            "n_measured",
            "n_computed",
            "ha_kA_m_median",
            "ha_T_median",
            "ha_kA_m_min",
            "ha_kA_m_max",
            "max_over_min",
            "n_rt",
            "median_basis",
            "has_hts",
            "sources",
        ]
    ]
    grouped[["ha_kA_m_median", "ha_kA_m_min", "ha_kA_m_max"]] = grouped[
        ["ha_kA_m_median", "ha_kA_m_min", "ha_kA_m_max"]
    ].round(2)
    grouped[["ha_T_median", "max_over_min"]] = grouped[["ha_T_median", "max_over_min"]].round(3)
    grouped = grouped.sort_values("n_records", ascending=False)

    conflicts = grouped[grouped["max_over_min"] >= 2.0].copy()
    conflicts["reason"] = "同配方 Ha 最大/最小 >= 2"
    conflicts = pd.concat([conflicts, pd.DataFrame(MANUAL_CONFLICTS)], ignore_index=True)
    log(
        logs,
        "aggregate",
        len(records),
        len(grouped),
        "按约化化学式聚合；max_over_min>=2 写入冲突表",
    )

    records.to_csv(out_dir / "33_ha_records.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(out_dir / "33_ha_dataset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(logs).to_csv(out_dir / "33_ha_data_log.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(out_dir / "33_ha_conflicts.csv", index=False, encoding="utf-8-sig")

    sync_dictionary()
    sync_index(out_dir)

    print(f"逐条记录 {len(records)} 条（可用 {len(usable)} 条），覆盖配方 {len(grouped)} 个")
    print(f"温度档：{usable['temp_band'].value_counts().to_dict()}")
    print(f"含 Dy/Tb 配方：{int(grouped['has_hts'].sum())} 个")
    print(f"冲突配方：{len(conflicts)} 个；待复核记录：{int(records['needs_review'].sum())} 条")
    report_validation(grouped, usable)
    print(f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")


def report_validation(grouped: pd.DataFrame, records: pd.DataFrame) -> None:
    """用已知体系对照与门槛覆盖率自检，数字只打印不写进 md。"""
    print("--- 已知体系对照 ---")
    for formula, literature_T in (("Nd2Fe14B", 6.7), ("SmCo5", 30.0)):
        hit = grouped[grouped["formula"].eq(formula)]
        if len(hit):
            value = float(hit["ha_T_median"].iloc[0])
            print(f"{formula}: 数据集 {value:.2f} T，文献量级 {literature_T:.1f} T，比值 {value / literature_T:.2f}")

    print("--- 门槛覆盖率（spec 5.3 要求 >=300 条）---")
    total = len(records)
    rt = int(records["temp_band"].eq("RT").sum())
    hts = int(grouped["has_hts"].sum())
    print(f"总记录 {total} 条（目标 300）：{total / 300:.0%}")
    print(f"室温档 {rt} 条；含 Dy/Tb 配方 {hts} 个（目标 100）")


def sync_index(out_dir: Path) -> None:
    """把输出文件的行列数补进 dataset_index.csv（幂等）。"""
    index_path = ROOT / "dataset/dataset_index.csv"
    if not index_path.exists():
        return
    table = pd.read_csv(index_path)
    entries = {
        "shared/processed/33_ha_records.csv": ("外禀/各向异性", "NEMAD 高矫顽力档 + 文献 Ha", "Ha 逐条记录，含出处、温度与证据句"),
        "shared/processed/33_ha_dataset.csv": ("外禀/各向异性", "33_ha_records.csv 聚合", "按配方聚合的 Ha 中位数与极差"),
        "shared/processed/33_ha_data_log.csv": ("外禀/各向异性", "33_ha_records.csv 处理过程", "Ha 数据每一步行数变化"),
        "shared/processed/33_ha_conflicts.csv": ("外禀/各向异性", "33_ha_records.csv 聚合", "同配方 Ha 分歧与已知抽取问题"),
    }
    table = table[~table["file"].isin(entries)]
    rows = []
    for name, (layer, source, purpose) in entries.items():
        path = ROOT / "dataset" / name
        sample = pd.read_csv(path)
        rows.append(
            {
                "file": name,
                "layer": layer,
                "rows": len(sample),
                "columns": sample.shape[1],
                "source": source,
                "purpose": purpose,
                "refresh_note": "python scripts/33_build_ha_dataset.py",
            }
        )
    table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(index_path, index=False, encoding="utf-8-sig")


def sync_dictionary() -> None:
    """把本脚本输出的字段补进 data_dictionary.csv（幂等）。"""
    path = ROOT / "dataset/data_dictionary.csv"
    table = pd.read_csv(path)
    files = {row[0] for row in DICT_ROWS}
    table = table[~table["file"].isin(files)]
    add = pd.DataFrame(DICT_ROWS, columns=["file", "column", "meaning", "unit_or_range"])
    table = pd.concat([table, add], ignore_index=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
