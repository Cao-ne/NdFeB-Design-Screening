"""抓取冻结材料 ID 的 Materials Project 摘要与完整结构。

只做原始数据归档，不做清洗、特征或建模。API key 从环境变量
``MP_API_KEY`` 读取，不写入任何输出文件。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://api.materialsproject.org/materials/summary/"
FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "elements",
    "nelements",
    "nsites",
    "symmetry",
    "density",
    "volume",
    "is_stable",
    "is_magnetic",
    "ordering",
    "energy_above_hull",
    "formation_energy_per_atom",
    "total_magnetization",
    "total_magnetization_normalized_formula_units",
    "total_magnetization_normalized_vol",
    "num_magnetic_sites",
    "num_unique_magnetic_sites",
    "types_of_magnetic_species",
    "theoretical",
    "deprecated",
    "last_updated",
    "structure",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, low_memory=False)
    if "material_id" not in frame.columns:
        raise ValueError(f"{path} 中没有 material_id 列")
    ids = frame["material_id"].astype(str).dropna().drop_duplicates().tolist()
    return ids


def fetch_batch(session: requests.Session, ids: list[str], api_key: str) -> dict:
    last_error = None
    for attempt in range(5):
        try:
            collected: list[dict] = []
            offset = 0
            last_meta: dict = {}
            while True:
                params = {
                    "material_ids": ",".join(ids),
                    "_fields": ",".join(FIELDS),
                    "_limit": 1000,
                    "_skip": offset,
                }
                response = session.get(
                    API_URL,
                    params=params,
                    headers={"X-API-KEY": api_key},
                    timeout=180,
                )
                if response.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    break
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") or []
                collected.extend(data)
                last_meta = payload.get("meta") or {}
                total = int(last_meta.get("total_doc") or 0)
                offset += len(data)
                if not data or offset >= total:
                    return {"data": collected, "meta": last_meta}
                time.sleep(0.15)
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"批次请求失败：{last_error}")


def flatten_summary(item: dict) -> dict:
    symmetry = item.get("symmetry") or {}
    elements = item.get("elements") or []
    magnetic_species = item.get("types_of_magnetic_species") or []
    return {
        "material_id": item.get("material_id"),
        "formula_pretty": item.get("formula_pretty"),
        "chemsys": item.get("chemsys"),
        "elements": ";".join(map(str, elements)),
        "nelements": item.get("nelements"),
        "nsites": item.get("nsites"),
        "space_group_symbol": symmetry.get("symbol"),
        "space_group_number": symmetry.get("number"),
        "crystal_system": symmetry.get("crystal_system"),
        "point_group": symmetry.get("point_group"),
        "density_g_cm3": item.get("density"),
        "volume_A3": item.get("volume"),
        "is_stable": item.get("is_stable"),
        "is_magnetic": item.get("is_magnetic"),
        "ordering": item.get("ordering"),
        "energy_above_hull_eV": item.get("energy_above_hull"),
        "formation_energy_per_atom_eV": item.get("formation_energy_per_atom"),
        "total_magnetization_uB": item.get("total_magnetization"),
        "moment_uB_per_fu": item.get("total_magnetization_normalized_formula_units"),
        "moment_uB_per_A3": item.get("total_magnetization_normalized_vol"),
        "n_magnetic_sites": item.get("num_magnetic_sites"),
        "n_unique_magnetic_sites": item.get("num_unique_magnetic_sites"),
        "magnetic_species": ";".join(map(str, magnetic_species)),
        "theoretical": item.get("theoretical"),
        "deprecated": item.get("deprecated"),
        "last_updated": item.get("last_updated"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取 MP 摘要与完整结构")
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("dataset/MP/raw/09_mp_magnetism_raw.csv"),
        help="包含 material_id 列的冻结 ID 表",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset/MP/raw"),
        help="输出目录",
    )
    parser.add_argument("--batch-size", type=int, default=300)
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY", "").strip()
    if not api_key:
        print("未设置 MP_API_KEY；请通过环境变量传入，不要把 key 写进脚本。")
        return 1

    ids = load_ids(args.ids)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = args.out_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    summaries: list[dict] = []
    structures: list[dict] = []
    missing: list[str] = []
    logs: list[dict] = []
    api_versions: set[str] = set()
    fetched_at = datetime.now(timezone.utc).isoformat()

    for start in range(0, len(ids), args.batch_size):
        batch = ids[start:start + args.batch_size]
        began = time.time()
        payload = fetch_batch(session, batch, api_key)
        data = payload.get("data") or []
        meta = payload.get("meta") or {}
        if meta.get("api_version"):
            api_versions.add(str(meta["api_version"]))

        returned = {str(item.get("material_id")) for item in data}
        batch_missing = sorted(set(batch) - returned)
        missing.extend(batch_missing)

        for item in data:
            material_id = str(item.get("material_id"))
            summaries.append(flatten_summary(item))
            structure = item.get("structure")
            if structure:
                structures.append({
                    "material_id": material_id,
                    "formula_pretty": item.get("formula_pretty"),
                    "structure": structure,
                })

        logs.append({
            "batch_index": start // args.batch_size,
            "requested": len(batch),
            "returned": len(data),
            "missing": len(batch_missing),
            "seconds": round(time.time() - began, 2),
            "status": "ok",
        })
        print(
            f"批次 {start // args.batch_size + 1}: "
            f"请求 {len(batch)}，返回 {len(data)}，缺失 {len(batch_missing)}"
        )

    summary_path = args.out_dir / "materials_summary.csv"
    pd.DataFrame(summaries).sort_values("material_id").to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )

    summary_jsonl = args.out_dir / "materials_summary.jsonl.gz"
    with gzip.open(summary_jsonl, "wt", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    structure_path = args.out_dir / "structures.jsonl.gz"
    with gzip.open(structure_path, "wt", encoding="utf-8") as handle:
        for row in structures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if missing:
        pd.DataFrame({"material_id": missing}).to_csv(
            args.out_dir / "missing_materials.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=["material_id"]).to_csv(
            args.out_dir / "missing_materials.csv", index=False, encoding="utf-8-sig"
        )

    pd.DataFrame(logs).to_csv(
        args.out_dir / "fetch_log.csv", index=False, encoding="utf-8-sig"
    )

    manifest_rows = []
    for path, description, rows in [
        (summary_path, "MP 摘要（扁平 CSV）", len(summaries)),
        (summary_jsonl, "MP 摘要（完整 JSONL）", len(summaries)),
        (structure_path, "MP 完整结构（含坐标）", len(structures)),
        (args.out_dir / "missing_materials.csv", "未返回的 material_id", len(missing)),
        (args.out_dir / "fetch_log.csv", "抓取批次日志", len(logs)),
    ]:
        manifest_rows.append({
            "file": path.name,
            "description": description,
            "rows": rows,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "source": "Materials Project API /materials/summary/",
            "retrieved_at_utc": fetched_at,
            "api_version": ";".join(sorted(api_versions)),
        })
    pd.DataFrame(manifest_rows).to_csv(
        args.out_dir / "source_manifest.csv", index=False, encoding="utf-8-sig"
    )

    print(f"写入 {summary_path}：{len(summaries)} 行")
    print(f"写入 {structure_path}：{len(structures)} 条结构")
    print(f"缺失 {len(missing)} 条")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
