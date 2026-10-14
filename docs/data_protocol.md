# 真实数据协议 v1

本协议于 2026-10-14 经用户确认规则后冻结。实现基于主干 `01d3cb8`，不改变模型、扩散过程、损失或概率指标公式。真实数据运行仍处于接口验收阶段。

## 数据来源与时钟

原始 CSV 来自用户明确指定的 `/home/u86177/work/multi_scaleCADM/Data`。仅读取原始表，不引用该项目的代码、切分、权重或结论。副本尚未与官方原始压缩包及转换过程核验，因此继续标为 `gefcom2014_format_candidate`。

采用 GEFCom2014 原任务的每日午夜起报、未来 `+1,…,+24` 小时约定，依据竞赛作者[论文 §6.1](https://robjhyndman.com/papers/gefcom2014.pdf)。本地 CSV 没有独立起报时间、实际可用时间或时区字段；这个时钟是明确的基准假设，不能据此证明副本的 NWP 发布延迟。时间保留无时区的原始时钟，不加 `Z`，不擅自转换成 UTC。

功率单位为额定容量比例；NWP 顺序固定为 `U10,V10,U100,V100`。日历为小时正余弦和年内进度正余弦；年内进度按实际年份长度计算，包含闰日。

## 风场选择与时间划分

配置见 [`configs/data_v1.json`](../configs/data_v1.json)。首先仅在下列训练候选区间评估完整日窗口比例，要求达到 100%，再选合格者中最小编号。程序实际得到：Zone 1 为 542/547，Zone 2 为 543/547，Zone 3–10 为 547/547，所以选定 **Zone 3**。没有使用验证／测试分数或完整率来选择风场。

按发布时间（每日 00:00，两个端点均包含）划分：

| 集合 | 发布时间范围 | 候选窗口 | 保留 | 丢弃 |
| --- | --- | ---: | ---: | ---: |
| Train | 2012-01-01 至 2013-06-30 | 547 | 547 | 0 |
| Validation | 2013-07-01 至 2013-09-30 | 92 | 71 | 21 |
| Test | 2013-10-01 至 2013-12-31 | 92 | 90 | 2 |

每个窗口的有效时间为 `issue_time + 1h ... +24h`。例如 2013-06-30 的最后一个目标时刻是 2013-07-01 00:00，验证首个窗口的第一个目标时刻是 2013-07-01 01:00；不重叠。程序逐个检查目标时间主键在窗口及集合之间不交叠。原文件名中的 Train/Test 不直接决定实验切分；表按主键连接后重新按协议划分。

这一测试时段只覆盖 2013 年最后三个月，不能支持全年泛化结论。当前测试集仅被准备与审计，没有用于评分或选检查点。

## 可重放的处理与数据身份

1. 记录所读取原文件的 SHA-256、大小；列名、zone 和整点时间必须正确。
2. 以 `(ZONEID,TIMESTAMP)` 解析与连接功率、NWP；文件内及跨文件重复主键均报错，不能按行号直接拼接。
3. 枚举协议内每一个发布时间，查找精确的 24 个小时；缺行、缺失／非有限数值或越界功率会丢弃整窗，并记录对应小时和原因。合法的 0、1 功率保留。
4. 仅在最终训练窗口拟合逐特征 NWP 均值和总体标准差（`correction=0`，下限 `1e-6`），验证／测试数据不参与拟合。
5. 每个样本保留 `sample_id、zone_id、issue_time、valid_times、split`，以及每小时功率和 NWP 的来源文件／CSV 行号。身份信息不输入模型。
6. 数据清单包含协议、原文件、样本顺序、丢弃记录、张量内容与标准化统计量的指纹。加载数据包会核对内容；真实训练检查点和批次迭代器也绑定这份指纹。

输入文件可以重排行，但样本仍按发布时间构造。重排后的字节哈希和来源行号不同，会形成新的数据版本；不能自动当作原检查点的续训数据。确认官方来源时仍可另按主键比较字段，物理哈希不承担来源真实性证明。

不含原始功率／气象记录的冻结清单见 [data_v1_data_manifest.json](data_v1_data_manifest.json)，全部丢弃样本及小时原因见 [data_v1_dropped_windows.json](data_v1_dropped_windows.json)。本次数据指纹：

```text
403ff1adc1df6b1bd6dd79b24f206f164daf3161f829bf7a232fddf90a9168a4
```

## 准备与追溯命令

在 `code_rl_clean` 环境、仓库根目录运行。所需 torch 和 NumPy 版本列于 `requirements.txt`；已有环境无需重装。

```bash
python prepare_data.py --source /home/u86177/work/multi_scaleCADM/Data \
  --config configs/data_v1.json --output outputs/data_v1_new

python prepare_data.py --source /home/u86177/work/multi_scaleCADM/Data \
  --package outputs/data_v1_new/data.pt --sample-id zone03_20120101T000000
```

准备程序拒绝覆盖已有输出，生成 `data.pt`、`data_manifest.json`、`sample_metadata.json`、`dropped_windows.json` 和 `trace_sample.json`。原始数据、张量和带原始数值的追溯文件均留在 `outputs/` 或外部数据目录，不提交版本管理。

示例样本 `zone03_20120101T000000` 对应 `Train_W_Zone3.csv` 的第 2–25 行：第 2 行为 2012-01-01 01:00（lead=1），第 25 行为 2012-01-02 00:00（lead=24）。追溯入口先验证源文件哈希，再返回全部 24 小时的原始记录。

## 带身份的训练与评价

`train_real.py` 仅运行 BASE-2，复用现有 Trainer、WeatherScaler、EMA 和 DDIM，不另写训练更新算法。保留原 `train.py` 合成入口。真实断点需要提供匹配的数据指纹；相同样本数量但内容不同也拒绝恢复。带故障标记的快照继续禁止续训。

```bash
python train_real.py --data outputs/data_v1_new/data.pt --device cuda \
  --updates 20 --validation-limit 4 --scenarios 16 --sampling-steps 10 \
  --validate-every 10 --output outputs/real_base_check

python train_real.py --data outputs/data_v1_new/data.pt --device cuda \
  --resume outputs/real_base_check/checkpoint_last.pt --updates 2 \
  --output outputs/real_base_resume
```

这里 `--updates` 表示新增成功更新数，跳步另计。恢复从检查点读取训练和评价配置，不以新的 CLI 默认值覆盖它们；上一运行目录中的 `checkpoint_best.pt` 也必须保留。`--validation-limit 4` 明确只用最早的 4 个有效验证日做接口检查；正式开发评价应移除此参数，使用全部 71 个有效验证日。默认采样为 100 条场景、50 步 DDIM。

运行产物包括 `config.json`、数据清单、last/best 检查点、训练与验证历史、每个验证节点的 raw/EMA 场景，以及独立 RAND 参照。RAND 只从训练集抽取完整 24 小时轨迹，保存抽样索引。验证使用独立随机数生成器，raw/EMA 复用同一初始噪声规则，不推进训练加噪或批次抽样的 RNG。

检查点固定按 **EMA 验证平均 CRPS 严格下降**选择，相同分数保留更早者，包含 update=0 的诊断节点。raw 权重评分只用于观察，不参与最后择优。运行根目录的场景和分数来自选中的 EMA 检查点。

新评价入口的 `scenarios.pt`、`targets.pt` 各自包含张量与独立身份信息，区别于旧 `metrics.py` 接受的裸张量。评价先按 `sample_id` 重排预测，再核对有效时间、风场、split、单位、时钟和数据／协议指纹；重复、缺失或额外样本立即报错。元信息不能证明人为标错标签的张量来源，因此生产端和包加载端同时保留身份与内容指纹。

```bash
python evaluate.py --scenarios outputs/real_base_check/scenarios.pt \
  --targets outputs/real_base_check/targets.pt --output outputs/replayed_scores
```

保存 `scores_per_window.pt` 中的 CRPS `[D,24]`、ES／VS `[D]`，以及每个覆盖率水平的 covered／width `[D,24]`，并保留对应 ID 顺序。汇总指标口径与 `metrics.py` 相同，没有更换有限样本估计方式。

当前尚未开展 1,000 次更新的学习性试验、全验证集模型选择或留出测试评价。接口检查不是循环机制收益的证据。
