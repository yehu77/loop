# 单风场条件循环 DiT

研究项目：根据 NWP 和日历生成单风场未来 24 小时的完整功率轨迹，比较浅层、共享循环和非共享加深的扩散去噪器。

**当前状态：已有实验均已归档，没有自动追加训练。** 三种子架构对照、R01–R04、历史输入对照和预测校正器均已完成。现有证据没有建立明确、稳定的共享循环质量—成本优势；测试集保持未评分。具体判断以每个实验的冻结协议和报告为准。

## 从这里开始

| 你要做什么 | 入口 |
|---|---|
| 查看实验进度、结论、配置与结果 | [实验目录](docs/experiments/README.md) |
| 跑通代码、检查环境和测试 | [快速上手](docs/guide/quickstart.md) |
| 找到模型、数据、训练与评价代码 | [代码地图](docs/guide/code-map.md) |
| 重评场景、理解断点和冻结版本 | [复现说明](docs/guide/reproduction.md) |
| 查找全部协议与历史验收 | [文档首页](docs/README.md) |
| 修改代码、提交实验记录 | [贡献说明](CONTRIBUTING.md) |

## 常用命令

在仓库根目录使用已有环境：

```bash
conda activate code_rl_clean
make help
make catalog                    # 实验目录；只读，不启动训练
make experiment ID=prediction-correction
make check                      # 检查导航引用及目录同步；无需 PyTorch
make test                       # CPU 回归测试
make smoke                      # 三模型 CPU 合成检查，结果写入新目录
```

没有 `make` 时可直接执行：

```bash
python tools/catalog.py list
python -m unittest discover -s tests -v
python train.py --variant all --device cpu --updates 3 --scenarios 4 --sampling-steps 4
```

模型代码使用 Python 3.10+、PyTorch 2.10。已有环境无需重装；合成检查只验证运行路径，不代表真实预测效果。详细参数、GPU 检查和数据准备见[快速上手](docs/guide/quickstart.md)。

## 仓库布局

```text
model.py / diffusion.py          模型与扩散过程
train.py / train_real.py         合成检查与真实数据训练
correction.py / train_correction.py  独立校正器原型
(data / history_data / metrics / evaluate).py  数据与评价
configs/                        已冻结的数据及实验配置
diagnostics/                   专项验收、实验队列与汇总
tools/                         仓库导航工具，不参与训练
tests/                         unittest 回归测试
docs/guide/                    当前使用说明
docs/experiments/README.md      全部实验的统一入口
docs/{pilots,diagnostics,experiments}/  原始归档证据
outputs/                       本地产物，不提交 Git
```

旧源码和实验目录被检查点及哈希清单引用，因此保留原路径；新导航集中呈现它们。[整理原则与参考项目](docs/guide/organization.md)说明了这一取舍。

## 数据与比较边界

- 原始协议：Zone 3，547 / 71 / 90 个训练／验证／测试窗口。
- 共同历史协议：545 / 57 / 89；采用明确记录的 1 小时观测延迟假设。
- 两套验证日期不同，不能直接拼成同条件排行榜。原始来源与实际 NWP 可用时间的证据边界见[数据协议](docs/data_protocol.md)。
- 数据、检查点和大张量保留在本地；GitHub 保存代码、配置、报告和审计清单。克隆仓库不会自动取得本机数据或检查点。
