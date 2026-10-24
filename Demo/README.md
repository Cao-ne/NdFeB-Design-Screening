# NdFeB 成分模型 Demo

输入化学式，输出 Ms、Tc、Hcj 的点值、80% 区间与适用域。三套 XGBoost，纯成分特征。

## 模型

| 任务 | 目标变换 | 单位 | 说明 |
| --- | --- | --- | --- |
| Ms | 无 | emu/g | 伪 Huber 损失 |
| Tc | 无 | K | 伪 Huber 损失 |
| Hcj | log10(1+Hcj) | kA/m | 展示时还原；只作区间与档位 |

80% 区间由折外残差校准；适用域由标准化成分空间的最近邻距离判定。

## 输入

当前：直接输化学式，如 Nd2Fe14B、Nd2Fe13.5Co0.5B。

待建：2:14:1 子晶格占有率界面，由 compose.py 生成化学式，见 [../spec.md](<../spec.md>) 2.1。

## 运行

    python Demo/Ms/train.py
    python Demo/Tc/train.py
    python Demo/Hcj/train.py
    python Demo/app.py

浏览器打开 http://127.0.0.1:5000。首次运行先装 [requirements-web.txt](<requirements-web.txt>)。

命令行预测：

    python Demo/Ms/predict.py Nd2Fe14B

## 产物

每个任务 output/ 下有：model.joblib、meta.json、fold_metrics.csv、oof.csv、feature_importance.csv、data_log.csv、conflicts.csv。

指标以 meta.json 为准，文档不复制。

## 数据边界

- raw 只读，同一还原化学式取标签中位数，冲突写入 conflicts.csv。
- 交叉验证按化学体系分组，避免同体系泄漏。
- Hcj 缺 Ku、晶粒尺寸、晶界与测量温度，只能粗筛。
- 模型在宽成分数据上训练，2:14:1 属外推，页面需给适用域提示。

## 当前缺口

成本、基准对比、推荐短名单、占有率输入界面，均见 [../tasks.md](<../tasks.md>) T02–T06。
