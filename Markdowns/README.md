# NdFeB 系稀土永磁体：成分—性能预测与低成本配方筛选

> 阶段：中期 Demo（2026-10-21）｜文中性能数字均为模型预测，非实测

## 1. 一句话

输入一个 NdFeB 系成分，预测 Ms、Tc、Hcj，估算原料成本，回答"少用重稀土，性能掉多少、成本省多少"。

## 2. 两个创新方向

| 方向 | 内容 | 落点 |
| --- | --- | --- |
| 一 | 成分比例 → 性能 | 输入做成 2:14:1 子晶格占有率（RE 位 2、Fe 位 14、B 位 1），"RE 位换 10% Ce"成为可拨的旋钮 |
| 二 | 便宜元素替代 | 给定性能下限，反搜最低成本成分，输出性能—成本帕累托前沿与短名单 |

两者是一条链：方向一定义输入，方向二用预测做搜索。不是两个并列课题。

## 3. 三个输出

| 输出 | 单位 | 回答什么 | 呈现方式 |
| --- | --- | --- | --- |
| Ms | emu/g | 磁体理论上能有多强 | 点值 + 80% 区间 + 适用域 |
| Tc | K | 能工作到多热 | 同上 |
| Hcj | kA/m | 反向磁场或高温下会不会退磁 | 点值 + 区间 + 档位 + 适用域 |

成本（美元/kg）是第四个决策轴，不是物理量；半定量，不含工艺与损耗。

## 4. 架构

    ① 成分定义   compose.py      占有率 → 化学式 + 校验（待建）
    ② 特征       featlib.py      54 维，训练与页面共用（已有）
    ③ 预测       Demo/Ms、Tc、Hcj  三套 XGBoost + 区间 + 适用域（已有）
    ④ 成本       cost.py         质量分数 × 元素单价（待建）
    ⑤ 搜索       search.py       性能下限 → 帕累托 → 短名单（待建）
    ⑥ 呈现       Demo/web        单页（已有，缺成本、对比、短名单）
    ⑦ Hcj 图线   graph_hcj.py    结构图 + 工艺 token → Hcj（路线 B，待建，未过硬门槛不进主结果）

Hcj 另有一条标签治理线，见第 5 节。

## 5. Hcj 为什么难

Hcj 是外禀量：由畴壁形核/钉扎、晶粒尺寸、晶界相决定，不由单位胞决定。现有数据三处硬伤（2026-10-21 核对，源文件 [hcj_cleaned.csv](<dataset/NEMAD/processed/hcj_cleaned.csv>)）：

| 问题 | 事实 |
| --- | --- |
| 测量温度混杂 | 1345 条中仅 132 条记温度；81 条 ≤50 K，250–350 K 只有 30 条 |
| 同成分工艺离散 | 有 ≥3 条重复的 40 个配方，同配方 max/min 中位数 5.3 倍，一半 >5 倍 |
| 目标体系为空 | 2:14:1 仅 18 条，其中含 Dy/Tb 的 0 条 |

结论：先治标签，再谈模型。路线见 [plan.md](<plan.md>) 阶段二。

## 6. 数据

| 来源 | 规模 | 用途 |
| --- | --- | --- |
| MP（DFT） | 4291 条，去多形体 3395 条 | 结构、稳定性、DFT 磁矩 |
| NEMAD（实验） | Ms、Tc、Hcj 三套 | 唯一的实验标签来源 |
| 文献抽取 | 65 条 + 367 行表格 | 补充与外部校验 |

已知问题（2026-10-21 复核）：

- [30_design_table.csv](<dataset/shared/processed/30_design_table.csv>) 104 行中含 Nd 的只有 24 行，不是纯 NdFeB。
- [18_features_all.csv](<dataset/MP/processed/18_features_all.csv>) 3395 行中 1795 行非闭合、1095 行氧化物。
- [31_search_shortlist.csv](<dataset/shared/processed/31_search_shortlist.csv>) Dy_share=0.5 但化学式无 Dy/Tb，不可用。
- 训练集 2834 个化学式中 2:14:1 仅 30 条：模型是"宽成分 RE-Fe-B 模型"，对 2:14:1 属外推。
- 旧脚本仍读写 data/，目前只有 Demo 能跑通。

## 7. 目录

    dataset/   MP、NEMAD、shared 三源，raw 只读，processed 存放清洗结果
    Demo/      本地网页与三套模型（唯一可端到端运行的部分）
    scripts/   抓取、清洗、特征、建模、筛选（路径待修复）
    docs/      大纲、申报表与报告模板
    Papers/    36 篇参考文献
    往年相似案例/

## 8. 文档

| 文件 | 管什么 |
| --- | --- |
| [spec.md](<spec.md>) | 需求与验收 |
| [plan.md](<plan.md>) | 三阶段排期 |
| [tasks.md](<tasks.md>) | 当前任务 |
| [checklist.md](<checklist.md>) | 汇报前逐项验收 |
| [AGENTS.md](<AGENTS.md>) | 协作规则 |
| [docs/大纲.md](<docs/大纲.md>) | 申报与成果叙事 |
| [Demo/README.md](<Demo/README.md>) | 运行说明 |

## 9. 跑起来

    python Demo/app.py
    浏览器打开 http://127.0.0.1:5000

首次运行前：python -m pip install -r Demo/requirements-web.txt

## 10. 边界

- 成分线不含原子坐标与工艺；Hcj 另有路线 B（结构图 + 工艺 token，[spec.md](<spec.md>) 5.5），未过门槛前不进主结果。
- Hcj 只给区间与档位，不做趋势强结论。
- 成本是原料估算，不等于企业采购价或降本。
- 推荐配方未经实验验证。
