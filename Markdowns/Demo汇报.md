# 基于深度学习的NdFeB系稀土永磁体相关性质预测

## 项目主线

1.基于三元永磁体，预测不同元素配比下的磁体性质

2.基于非NdFeB系永磁体，减少元素/增加元素预测磁体性质

3.目的是有实用价值，能为企业降本增效

## 有关项目的基本说明

### 预测的物理量：

矫顽力

居里温度

饱和磁化强度

原因：三者为论文里的常用口径

### 数据集

1.饱和磁化强度

来源：NEMAD，MP(理论计算值)

处理要点：

1. **目标量选 Ms\_emu\_g，不选 μ0Ms**。emu/g 可以直接测量，不需要密度；从 emu/g 换算到 μ0Ms 需要密度，而密度混算在本项目 104 条 DFT 密度上中位误差为 10%，不足以支持这一步换算。因此，`need_density`（234 行）和 `ambiguous`（30 行）被剔除。
2. **同一材料多次测量取中位数**，聚合后被模型使用 **1682 条**。
3. **MP 侧去多形体**：同一化学式只保留凸包能最低的条目，4291 条变为 3395 条，剔除 **903 条**，逐条留痕于 **14\_polymorph\_removal\_log.csv**（15 列，含保留/剔除的 Ms\_T 与 hull 能差）。按化学体系分组切分后冻结：train 2375 / val 510 / test 510（**15\_split\_table.csv**）。
4. **特征层**：**18\_features\_all.csv** 3395×103，包含成分分数、化学计量比、加权描述符、4f 磁性先验和成本指数；并保留 `is_oxide` 1095 行、`flagged_nonclosure` 1795 行标记。
5. **DFT 标签治理**：**29\_ms\_label\_conflict.csv**（12×9）对 2:14:1 家族逐组分比对 DFT 与实验 μB/f.u.，发现重稀土 2:14:1 的 MP 总磁矩没有计算 4f-Fe 反平行耦合。合并表里由此产生 `ms_label_status`：`dft_ok` 100、`dft_suspect` 4。
6. **跨来源校验**：实验模型预测设计空间 77 个组分（**28\_ms\_imputed\_design\_space.csv** 77×13），再与 DFT 值比较；104 行中有 77 行可以得到成分法 Ms。
7. 文献侧另有 **03\_literature\_records\_raw\.csv** 中 **29 条** Ms，经透视进入 **08\_literature\_wide\_processed.csv**；Ms\_T 17 个非空、Ms\_emu\_g 仅 1 个非空，未进主训练。

<br />

2.居里温度

来源：NEMAD

处理要点：

1. **无 DFT 源**。MP 与 NEMAD 都不提供 Tc，因此 Tc 只有实验标签。数据里 `tc_method` 为 `exp` 1215 条、`unknown` 1732 条，未做来源剔除，只记标记。
2. **去重与剔除**：按化学式取中位数，剔除 6 条已知错误值（Nd2Fe14B 450 与 768.15、Nd2Fe17 743、Ce2Fe14B 490、Er2Fe14B 314、Y2Co17 940），剔除 `tc_K < 10` 的哨兵值，再剔除无稀土或特征缺失的行，得到 **1958 条**进模型。
3. **特征与 Ms 完全同构**，只用化学式可推的描述符，刻意不用密度、凸包能这类 DFT 专属量，避免不可迁移。
4. **真实留出检验**：设计空间 23 个有实验 Tc 的组分不进训练，只作检验（**23\_tc\_model\_metrics.csv**）。
5. **补全分布**：设计空间 104 行中 23 行实验值、81 行模型预测，`tc_source` 列逐行标记。

<br />

3.饱和磁化强度

来源：NEMAD

处理要点：

1. **单位统一到 kA/m**（同时保留 `hcj_kOe`），并保留 `meas_temp_K` 字段。
2. **按标签质量分三档**（`hcj_value_class`）：
   - `plausible_hcj`（≤20 kOe）**1787** 条，作为主训练集。
   - `borderline`（20–60 kOe）**160** 条，从主模型剔除，只做纳入对照（1058 行口径）。
   - `likely_anisotropy_field`（>60 kOe）**77** 条，实为各向异性场 Ha，**一律剔除**。
3. **目标取 log10(1+Hcj)**，因为矫顽力跨约三个数量级，原始尺度下平方误差会被少数大值主导。
4. **按 material\_name 取中位数聚合**，再按特征可用性过滤，**974 条**进入模型（不是 1787 条）。子群：RE-Fe-B 152 行、含 Fe 稀土 623 行。
5. **测量温度缺失 1779/2024**，这一列保留但未参与筛选。
6. **适用域**：以训练集自身留一最近邻距离的 95 分位为阈值，输出 `ad_dist` / `in_domain`（104 行中 67 行在域内）、`hard_family`（由空间群推 2:14:1 / 2:17 / 1:12 / 1:5）、`hcj_trustworthy`（在域内且属已知硬磁家族，**38 行**）、`hcj_label_in_train`。
7. **文献侧未进标签**：**03\_literature\_records\_raw\.csv** 中仅 2 条 `coercivity_Hc`、2 条 `anisotropy_field_Ha`；**05\_literature\_tables\_raw\.csv** 367×11 是表格级原始 cells（带 `is_magnetic_table`/`magnetic_units` 标记），尚未结构化成 Hcj 标签。

## Demo描述

三个任务在 Demo 里**共用同一套代码**：**Demo/common/data.py** 负责读数据并生成特征，**Demo/common/model.py** 负责训练、校准和保存模型；每个任务只露出一份 `data.py` 和 `train.py`，其中的差异全部收敛成一份 `TASK` 配置字典。

**特征（54 维，三个任务完全相同，统一由 featlib.py 生成）**

- 26 个元素原子分数 + `frac_other`（`frac_other` 是单一伪物种）
- 9 个化学计量比值：`re_over_metal`、`b_over_re`、`fe_over_re`、`co_over_tm`、`hre_over_re`、`lre_over_re` 等
- 8 个加权描述符：`w_mean_z/mass/radius/electroneg/valence`、`mix_entropy`、`size_mismatch`、`electroneg_spread`
- 5 个 4f 磁性先验：`re_avg_degennes`、`re_avg_moment`、`tm_avg_moment`、`has_re`、`has_hre`
- 4 个标志位或质量量：`hre_mass_frac`、`frac_sum_known`、`untracked_frac`、`is_oxide`、`is_carbide`

**标签处理（三个任务相同）**

1. 化学式经 pymatgen 还原成化学式（`reduced_formula`）
2. 同一化学式有多条测量记录时，取**中位数**，同时保留 `y_min`、`y_max`、`n_records`、`y_spread`
3. 极差 >0 的化学式写进 `conflicts.csv`（Ms 331 条、Tc 136 条、Hcj 117 条）
4. 每次行数变化都写进 `data_log.csv`（before/after/removed/note）

**模型与验证（三个任务相同）**

- 统一用一个 `XGBRegressor`（`tree_method=hist`，`random_state=42`）。Demo 已从 scripts/ 层的 GBM/RF 对照换成单一的 XGBoost
- **GroupKFold 5 折，按化学体系分组**（元素集合相同的样本不会被分到不同折），避免同体系中换了个配比就算出新样本
- 每折先对目标做 z-score 标准化再训练，用验证集早停（Ms 80 轮，Tc/Hcj 60 轮），`best_iteration` 记入 `fold_metrics.csv`
- 最终模型用各折 `best_iteration` 的中位数重训全量，`n_estimators` 落到 `meta.json`

**两个后处理（三个任务相同，这是 Demo 相对 scripts/ 层新增的部分）**

- **80% 区间**：取折外残差绝对值的 80% 分位，作为固定半宽，随 `model.joblib` 保存；预测时 `pred ± half_width`，再经逆变换还原单位
- **适用域**：对成分矩阵做中位数填补和标准化，以样本自身留一最近邻距离的 95% 分位为阈值；预测时取到最近训练样本的距离，超阈值即标“域外”

## 要解决的问题

Hcj与制造工艺高度相关，无法通过传统机器学习方法进行有效预测\
解决方案：在后续使用图神经网络结合第一性原理进行理论计算
