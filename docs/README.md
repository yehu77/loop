# 文档首页

[返回仓库首页](../README.md) · [全部实验与结论](experiments/README.md)

## 日常使用

- [快速上手](guide/quickstart.md)：环境、CPU 检查、GPU 检查、命令入口。
- [代码地图](guide/code-map.md)：按数据流定位文件，不必逐个翻脚本。
- [复现说明](guide/reproduction.md)：读取旧结果、独立重评、正常恢复和源码冻结。
- [整理原则](guide/organization.md)：目录约定、开源参考、本次整理的边界。
- [贡献说明](../CONTRIBUTING.md)：新增实验与日常代码改动。

## 数据与评价约定

| 文档 | 内容 |
|---|---|
| [候选数据审计](data_audit.md) | 本地副本、缺失与对齐证据 |
| [原始数据协议](data_protocol.md) | Zone 选择、发布时间、547/71/90 切分 |
| [共同历史协议](history_information_v1.md) | 一小时延迟假设、545/57/89 共同窗口 |
| [概率评分定义](metrics.md) | CRPS、ES、VS、覆盖率和宽度 |
| [训练协议 v2](training_protocol_v2.md) | 延迟 EMA、共同初始化与固定终点评价 |

## 查实验结果

统一使用[实验目录](experiments/README.md)，其中每项都链接到报告、冻结配置、归档证据和本地运行路径。

历史报告中的“下一阶段未开始”等语句描述的是**写作当时**的状态，保留为历史记录。当前研究状态由实验目录汇总；不要从某份旧报告的进度段推断后续实验仍未完成。

尤其注意：71 日的原始任务与 57 日的历史共同窗口是不同评价面板；单种子与三种子也不能混称为同一证据强度。

## 原始工程验收与早期协议

- [最初的合成数据 / RTX 2060 验收](acceptance.md)
- [真实数据连接验收](real_data_acceptance.md)
- [v2 seed-42 架构对照](architecture_v2_s42.md)
- [v2 三种子实施协议](architecture_v2_multiseed_protocol.md)

验收数字属于各报告注明的环境和执行日期，不代表每次文档整理都重新运行了 GPU 测试。

## 归档目录怎么读

| 目录 | 保存内容 |
|---|---|
| `docs/data/` | 数据审计、共同窗口、来源清单 |
| `docs/pilots/` | BASE pilot、EMA 和早期架构对照 |
| `docs/diagnostics/` | R01–R04 的验收与结果 |
| `docs/experiments/` | 历史实验、校正器实验及统一导航 |
| `configs/` | 数据、训练、评价和源码身份的冻结设置 |
| `outputs/`（本地） | 完整数据包、检查点、场景、日志和短程验收 |

先读报告，再按需进入对应归档；无需从数百个 JSON/CSV 中寻找实验入口。
