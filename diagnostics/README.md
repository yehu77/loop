# 专项脚本导航

[代码地图](../docs/guide/code-map.md) · [实验目录](../docs/experiments/README.md)

本目录同时含只读分析和训练队列。**文件名含 `run_` 的入口可能启动多次训练**；已有实验均已归档，
日常查看结果用 `python tools/catalog.py list`，不需要启动队列。

| 阶段 | 验收 / 分析 / 汇总 | 会启动训练的队列 |
|---|---|---|
| R01 时间配对 | `temporal_dependence.py` | 无 |
| R02 LOCAL | `summarize_local_reference.py`、`plot_local_reference.py` | `run_local_reference.py` |
| R03 状态利用 | `loop_state_usage.py`、`plot_loop_state_usage.py` | 无 |
| R04 中间监督 | `summarize_intermediate_supervision.py`、`plot_intermediate_supervision.py` | `run_intermediate_supervision.py`、`resume_intermediate_supervision.py` |
| 历史 BASE | `check_history_interface.py`、`summarize_history_information.py`、`plot_history_information.py` | `run_history_information.py` |
| 历史 × 架构 | `check_history_architecture.py`、`summarize_history_architecture.py` | `run_history_architecture.py` |
| 预测校正器 | `check_prediction_correction.py`、`evaluate_prediction_correction.py` | 独立入口：根目录 `train_correction.py` |

`check_*` 中包含短程训练／续训验收，不全是只读操作。`evaluate_prediction_correction.py` 会重新采样；
只想重评已有场景时使用根目录 `evaluate.py`。`summarize_*` 一般读取保存产物，但会创建报告目录。

从仓库根目录以模块方式运行，例如 `python -m diagnostics.loop_state_usage --help`。
每个脚本的参数、冻结实现版本及产物身份，见对应实验报告。历史源码保护失败时先核对版本，
不要直接改哈希或移走旧目录让命令继续。
