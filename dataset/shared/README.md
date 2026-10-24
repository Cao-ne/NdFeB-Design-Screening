# 共享数据

NOMAD、文献、元素价格，以及 MP 与 NEMAD 的跨源结果。

各向异性场 Ha 数据集在 `processed/33_ha_*.csv`，由 `scripts/33_build_ha_dataset.py` 生成；
其中 NEMAD 一档为上游 QC 判定的推断标签（`hcj_value_class=likely_anisotropy_field`），标注 `needs_review`，未经逐条文献复核前不作实测使用。
