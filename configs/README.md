# 配置导航

[仓库首页](../README.md) · [实验目录](../docs/experiments/README.md)

这里存放冻结协议，不是可以批量重命名或统一格式化的普通样例。部分文件包含源码 SHA256、
数据指纹、原检查点身份和固定输出目录；改变字节或路径可能影响恢复与来源校验。

| 用途 | 配置 | 对应说明 |
|---|---|---|
| 原始数据 | [data_v1.json](data_v1.json) | [时间切分与数据协议](../docs/data_protocol.md) |
| 三种子架构运行清单 | [architecture_v2_runs.json](architecture_v2_runs.json) | [三种子结果](../docs/architecture_v2_three_seeds.md) |
| R01 时间配对 | [diagnostics/temporal_dependence_v1.json](diagnostics/temporal_dependence_v1.json) | [R01](../docs/temporal_dependence_v1.md) |
| R02 逐小时参照 | [local_attention_v1.json](local_attention_v1.json) | [R02](../docs/local_attention_v1.md) |
| R03 状态干预 | [diagnostics/loop_state_usage_v1.json](diagnostics/loop_state_usage_v1.json) | [R03](../docs/loop_state_usage_v1.md) |
| R04 中间监督 | [intermediate_supervision_v1.json](intermediate_supervision_v1.json)、[续训清单](intermediate_supervision_v1_resumed.json) | [R04](../docs/intermediate_supervision_v1.md) |
| 历史数据准备 | [history_information_v1.json](history_information_v1.json) | [历史协议](../docs/history_information_v1.md) |
| H0/H1/H24 BASE | [共同计划](history_information_v1/common.json)、[H0](history_information_v1/H0.json)、[H1](history_information_v1/H1.json)、[H24](history_information_v1/H24.json)、[执行冻结](history_information_v1/execution.json) | [筛查结果](../docs/history_base_screening_v1.md) |
| 历史 × 架构 | [history_architecture_v1.json](history_architecture_v1.json) | [六格结果](../docs/history_architecture_v1.md) |
| 预测校正器 | [prediction_correction_v1.json](prediction_correction_v1.json) | [K0/K1/K2 结果](../docs/prediction_correction_v1.md) |

早期 pilot 的实际运行配置在 `docs/pilots/` 对应归档里；统一定位入口是实验目录。
新研究应新增独立配置、输出目录和实验条目，不能用新设置覆盖旧结果的身份。
