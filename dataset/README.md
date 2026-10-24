# 数据集目录说明

## 结构

```text
dataset/
├─ MP/
│  ├─ raw/           MP 摘要、完整结构与原始磁性表
│  └─ processed/     MP 清洗、设计空间与基线结果
├─ NEMAD/
│  ├─ raw/           NEMAD 原始 Ms/Hcj 数据
│  └─ processed/     Ms/Tc/Hcj 清洗与训练结果
├─ shared/
│  ├─ raw/           NOMAD、文献、元素价格
│  └─ processed/     MP+NEMAD 跨源合并与筛选结果
├─ dataset_index.csv
└─ data_dictionary.csv
```

## 规则

- `raw/` 只读，保留原始列名、单位与来源；不原地修改。
- `processed/` 由脚本生成，可重跑；清洗、去重、标签冲突都要留日志。
- MP 原始数据为 4291 条摘要与 4291 条完整结构，结构含晶格、空间群、位点和坐标。
- NEMAD 原始文件与当前模型输入分别放在 `NEMAD/raw/` 和 `NEMAD/processed/`。
- 文献、NOMAD、元素价格，以及 MP+NEMAD 合并结果放在 `shared/`，避免被误标为单一来源。

完整文件、行列数、来源和校验值以 `dataset_index.csv` 与各目录的 `source_manifest.csv` 为准；字段含义见 `data_dictionary.csv`。

现有部分脚本仍使用旧的 `data/` 路径，重跑前需按新目录调整。
