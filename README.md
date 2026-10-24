# NdFeB-Design-Screening

NdFeB 系稀土永磁体的成分—性能预测与低成本配方筛选原型（大创项目）。

输入一个 NdFeB 系成分，预测 Ms（emu/g）、Tc（K）、Hcj（kA/m），估算原料成本（美元/kg，半定量），
回答"少用重稀土，性能掉多少、成本省多少"。

> 当前阶段：中期 Demo。所有性能数字均为**模型预测值**，非实测；推荐配方未经实验验证。

## 快速开始

```bash
python -m pip install -r Demo/requirements-web.txt
python Demo/app.py
# 浏览器打开 http://127.0.0.1:5000
```

## 目录

| 目录 | 内容 |
| --- | --- |
| `Demo/` | 本地网页与 Ms / Tc / Hcj 三套模型，目前唯一可端到端运行的部分 |
| `dataset/` | MP（DFT）、NEMAD（实验）、shared 三源数据，`raw/` 只读，`processed/` 存清洗结果 |
| `scripts/` | 抓取、清洗、特征、建模、筛选脚本（部分路径待修复） |
| `Markdowns/` | 需求、排期、验收清单与项目说明 |
| `Papers/` | 参考文献索引（PDF 不入库，见 `Papers/_fetch_log.csv`） |

## 文档

| 文件 | 管什么 |
| --- | --- |
| [Markdowns/README.md](Markdowns/README.md) | 项目总览 |
| [Markdowns/spec.md](Markdowns/spec.md) | 需求与验收 |
| [Markdowns/plan.md](Markdowns/plan.md) | 三阶段排期 |
| [Markdowns/tasks.md](Markdowns/tasks.md) | 当前任务 |
| [Markdowns/checklist.md](Markdowns/checklist.md) | 汇报前逐项验收 |
| [Markdowns/AGENTS.md](Markdowns/AGENTS.md) | 协作规则与红线 |
| [Demo/README.md](Demo/README.md) | 运行说明 |

## 不入库的内容

`.gitignore` 明确排除以下几类，只保留在本地：

- `docs/`：申报表、评审指标等校内材料（含个人信息）
- `Papers/*.pdf`：第三方文献全文（体积与版权原因）
- `*.joblib`：模型权重，可由 `Demo/*/train.py` 重新训练生成
- `scripts/tmp/`：临时下载分片

## 边界

- 成分线不含原子坐标与工艺；Hcj 的图线路线尚未过硬门槛，不进主结果。
- Hcj 只给区间与档位，不做趋势强结论。
- 成本是原料估算，不等于企业采购价或降本。
- 训练集 2834 个化学式中 2:14:1 仅 30 条，模型对 2:14:1 属外推。
