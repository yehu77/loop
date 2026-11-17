# 快速上手

[文档首页](../README.md) · [代码地图](code-map.md) · [复现说明](reproduction.md)

## 选择 Python

在仓库根目录运行命令。原本机使用 `code_rl_clean`：

```bash
conda activate code_rl_clean
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

模型代码需要 Python 3.10+、PyTorch 2.10；[requirements.txt](../../requirements.txt)列出项目依赖。已有可用环境时先复用，不要为了整理仓库重新安装。其他机器需要自行准备兼容环境和数据；仓库不提供自动下载的训练数据。

本机也可直接指定解释器：

```bash
make test PYTHON=/home/u86177/miniconda3/envs/code_rl_clean/bin/python
```

这条绝对路径是本机示例，不是跨机器要求。独立环境报告脚本 `check_wind_env.py` 支持 Python 3.8+，未安装 PyTorch 时也可以生成报告：

```bash
python check_wind_env.py
```

报告在 `outputs/environment/`。退出码 0 表示报告写入成功；请另看 `CUDA_BASIC_OK` 等状态，不能把它等同于完整模型的 GPU 验收。

## 先做小检查

```bash
make check       # 标准库即可，检查导航路径和实验索引
make test        # 全套 unittest，明确禁用 CUDA
make smoke       # 3 个模型各 3 次 CPU 合成更新与小规模采样
```

等价 Python 命令：

```bash
python tools/catalog.py check
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
python train.py --variant all --device cpu --updates 3 --scenarios 4 --sampling-steps 4
```

合成检查默认创建 `outputs/synthetic/<时间戳>/<模型>/`，保存 `checkpoint.pt`、`scenarios.pt`、`report.json`。结果是运行验收，不能用于判断循环的风电收益。

需要实际 GPU 验收时，在能访问 CUDA 的环境执行：

```bash
python train.py --variant all --device cuda
python train.py --variant loop --device cuda --amp
```

这些是合成检查，不是重启已完成的真实实验。沙箱的 CUDA 可见性与主机可能不同；CPU 结果不能认证 GPU 显存、吞吐或 AMP。

## 找到一个实验

```bash
python tools/catalog.py list
python tools/catalog.py show prediction-correction --local
python tools/catalog.py list --local
```

`--local` 只检查目录是否存在，不读取检查点、不执行训练，也不证明目录内容完整。实验状态来自归档报告，不是 GPU 任务监控。

正式命令在每项实验报告中。旧队列包含冻结路径和来源检查，已有输出目录时会拒绝覆盖；源码身份已改变时应使用对应原版本，而不是关闭保护继续运行。

## 数据准备入口

原始协议与历史协议是两种不同的数据包：

```bash
python prepare_data.py --help
python prepare_history.py --help
python train_real.py --help
python train_correction.py --help
python evaluate.py --help
```

原始数据准备示例（`SOURCE_DIR` 换成已核验的 CSV 目录，输出路径须全新）：

```bash
python prepare_data.py --source SOURCE_DIR --output outputs/data_review_new
```

历史数据还需要父数据包、原始来源及明确的历史协议。字段和时间对齐以[原始数据协议](../data_protocol.md)及[历史协议](../history_information_v1.md)为准，不从文件名猜测训练／测试身份。
