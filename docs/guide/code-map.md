# 代码地图

[文档首页](../README.md) · [实验目录](../experiments/README.md)

## 核心数据流

```mermaid
flowchart LR
    CSV[原始 CSV] --> Prep[prepare_data.py / prepare_history.py]
    Prep --> Data[身份绑定的数据包]
    Data --> Train[train_real.py]
    Train --> Core[model.py + train.py]
    Core --> Sample[diffusion.py]
    Sample --> Artifacts[带样本身份的场景与目标]
    Artifacts --> Eval[evaluate.py + metrics.py]
    Eval --> Reports[实验报告与逐日评分]
```

## 按职责定位

| 职责 | 入口 / 实现 | 读代码时关注什么 |
|---|---|---|
| 模型结构 | [model.py](../../model.py) | BASE / LOOP / UNTIED、共享范围、初始化、历史 mask、最终读出 |
| 扩散与采样 | [diffusion.py](../../diffusion.py) | 已开平方的 a/σ、DDIM 裁剪、条件缓存与场景分块 |
| 通用训练与合成检查 | [train.py](../../train.py) | 标准化器、独立 RNG、EMA、正常／故障检查点 |
| 真实训练入口 | [train_real.py](../../train_real.py) | 数据身份、验证采样、固定终点与辅助 best |
| 原始数据 | [data.py](../../data.py)、[prepare_data.py](../../prepare_data.py) | CSV 主键、窗口、发布时间切分与训练统计量 |
| 共同历史数据 | [history_data.py](../../history_data.py)、[prepare_history.py](../../prepare_history.py) | 可用时间假设、共同窗口、H0/H1/H24 可见性 |
| 评分 | [metrics.py](../../metrics.py)、[evaluate.py](../../evaluate.py) | 数值定义与 sample_id 对齐分别负责什么 |
| 校正器 | [correction.py](../../correction.py)、[train_correction.py](../../train_correction.py) | 冻结 D、只训练 C、预测空间反馈、独立训练时钟 |
| EMA 旧诊断 | [diagnose_ema.py](../../diagnose_ema.py) | 只读重构，不恢复旧优化器 |
| 分段运行汇总 | [summarize_runs.py](../../summarize_runs.py) | 续训链路、共同节点与配对结果 |
| 专项工具 | [diagnostics/README.md](../../diagnostics/README.md) | 按研究阶段查验收、队列和汇总脚本 |
| 仓库导航 | [tools/catalog.py](../../tools/catalog.py) | 只读实验目录，与训练代码无依赖关系 |

## 三种主体不是三个训练器

`base` 执行一个 Core 一次；`loop` 复用 Core；`untied` 执行独立 Core。
每个 Core 含输入再注入 adapter 和两个 AdaLN blocks。三种架构共用真实训练入口。
校正器是另一个明确隔离的实验：冻结 BASE 提案，训练独立 C；其 K 不等于模型的 `rounds`。

## 训练状态与配置边界

- 旧默认训练语义仍是 v1；v2 延迟 EMA 必须显式配置或从匹配断点恢复。
- `train.py --updates` 是追加的优化器更新**尝试**；`train_real.py` 和 `train_correction.py` 是追加的**成功更新**。不要跨入口混用含义。
- `configs/` 的源码哈希绑定的是实验实现，不是普通可随手整理的设置文件。
- `docs/` 下的历史复现脚本属于归档证据；开发时先寻找 `diagnostics/` 中相应实现。

根目录源码保留原址是为了兼容已有导入、执行命令、源码哈希和检查点。今后若迁移为包结构，应作为单独的、可回放验证的代码重构，不夹带到实验结果归档中。
