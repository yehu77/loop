# 读取结果、重评与续训

[文档首页](../README.md) · [实验目录](../experiments/README.md)

## 先区分你要做的事

| 目标 | 所需内容 | 是否训练 |
|---|---|---|
| 阅读结论 | 报告和 Git 中的 JSON/CSV | 否 |
| 重评已有场景 | 本地 `scenarios.pt`、`targets.pt` 和匹配评价实现 | 否 |
| 重新生成场景 | 权重、数据、来源记录、冻结采样配置与实现 | 否，但运行前向采样 |
| 从正常断点续训 | 完整优化器/EMA/RNG/数据状态及匹配源码 | 是，需要明确新预算 |
| 从头复现实验 | 对应源码版本、冻结协议、原数据和新输出目录 | 是 |

## 先查路径

```bash
python tools/catalog.py show prediction-correction --local
```

原始任务的数据包在本机 `outputs/data_v1_20261014/data.pt`；共同历史数据包在
`outputs/history_information_v1_20261103/data.pt`。它们的验证集分别是 71 日和 57 日。
Git 克隆中没有这些数据包或检查点，目录缺失不是安装包坏了。

## 独立重评保存场景

在匹配评价实现下，下面命令只读取预测和目标，在**全新**目录保存评分：

```bash
python evaluate.py \
  --scenarios outputs/prediction_correction_v1_s42_fixed/K0/scenarios.pt \
  --targets outputs/prediction_correction_v1_s42_fixed/K0/targets.pt \
  --output outputs/review_correction_k0_new
```

入口会核对 sample_id、风场、发布时间、有效时间、单位和数据指纹，而不是只比较张量形状。
CRPS/ES/VS/区间结果及逐窗口分数由该入口保存；增量和日均 CRPS 则由相应专项汇总脚本计算。

若要求**逐位**重放旧 CUDA 场景，应同时保留：权重、GPU 数值环境、完整条件编码形状、
完整初始噪声生成形状、采样网格及 chunk 划分。仅相同 seed 不保证裁切后的面板与全量计算逐位一致。

## 不要直接重跑旧队列

历史队列的参数、路径、源码及数据身份已冻结，许多输出目录也已存在。目录索引只提供导航，
不会放宽旧脚本的检查。需要从头复现时，先阅读报告中的实现提交和独立输出约定；必要时用单独
checkout/worktree 保留相应版本。不要删除旧产物、修改源哈希或改变选择规则来让命令通过。

完整训练器的 `checkpoint_last.pt` 和失败时的 `failure.pt` 含义不同：后者可能已经消费微批次但未
完成优化器更新，无法当作精确续训点。改文件名或清掉 failure 标记都不能恢复丢失的累积梯度。

## 新实验如何归档

1. 先冻结研究问题、信息集、数据面板、种子、训练预算、主评价和停止条件。
2. 新建独立配置、运行目录；需要续训时记录分段清单、起点哈希及追加更新数。
3. 保存完整原始产物到 `outputs/`；不要在 Git 中放数据包、检查点和场景大张量。
4. 将报告、关键配置、逐日评分、运行来源与审计清单放入 `docs/experiments/<id>/`。
5. 更新 `docs/experiments/catalog.json`，重建实验目录并运行 `make check`。

文件策略：新增实验尽量集中在一个归档目录；历史 `pilots/` 和 `diagnostics/` 已被引用，保留原址。
大而重复的元数据优先留本地并在归档中记录身份与路径，避免每次结果提交重复整套样本记录。
已存在的归档不会因这条新约定被删减或重写。
