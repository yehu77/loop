# 单风场条件循环 DiT

研究目标是在每次扩散去噪调用内部复用 Transformer 核心，生成单风场完整未来功率轨迹。已实现 v1 三个对照模型、扩散训练、EMA、断点恢复与 DDIM 采样；当前仅做合成数据工程检查，尚未接入真实数据或启动正式实验。

## v1 架构

```text
NWP + 日历 → 逐小时 MLP → C
带噪功率 xt + C → 投影 + 固定位置编码 → E
扩散步 t → 正弦编码 + MLP → τ

h = E
每次 Core 执行：
    h = Adapter(h, E)
    h = AdaLN Block 1(h, τ)
    h = AdaLN Block 2(h, τ)
最后一次执行后：h + τ → 噪声输出头
```

| 命令中的名称 | 模型 | Core 执行方式 | 默认参数量 |
| --- | --- | --- | ---: |
| `base` | BASE-2 | 一个 Core 执行一次 | 208,065 |
| `loop` | LOOP-2×2 | 同一个 Core 执行两次 | 208,065 |
| `untied` | UNTIED-4 | 两个独立 Core 各执行一次 | 365,697 |

默认 `H=24, F=4, d=64, heads=4, FFN=256`。每个小时一个 token；使用非因果注意力、无仿射 LayerNorm 和完整反向传播。Adapter 初始为 `[I, 0]`，AdaLN 调制层与最终输出线性层零初始化。循环内不改变 `E`、扩散时间或带噪输入，不创建新模块、不截断梯度。`untied` 的每个 Core 同样读取 `E`。

## 运行合成数据检查

在仓库根目录使用已验证的环境：

```bash
conda activate code_rl_clean
python train.py --variant all --device cuda
# 单独检查循环模型的 FP16 混合精度训练
python train.py --variant loop --device cuda --amp
# 无 GPU 时的小规模 CPU 检查
python train.py --variant all --device cpu --updates 3 --scenarios 4 --sampling-steps 4
```

默认每个模型执行 6 次优化器更新尝试，微批次 4、累积 8 次，有效批次 32。使用 AdamW（学习率 `1e-4`、矩阵权重衰减 `1e-4`、偏置不衰减）、梯度裁剪 1、EMA 0.999。仅预测噪声，损失为 FP32 MSE。AMP 溢出时同时跳过参数与 EMA 更新。

扩散采用 1000 步线性 beta（`1e-4 → 0.02`），每条轨迹共享一个随机噪声级。EMA 模型默认对两个保留的合成条件分别生成 100 条轨迹，执行 50 步 FP32 DDIM（`eta=0`），场景分批大小为 16。仅裁剪干净预测，并重算一致噪声方向；报告每个采样步裁剪前越界率及最终零／满功率占比。

合成数据共 128 个窗口，前 96 个用于训练和 NWP 标准化统计量。日历使用小时与年内进度的正余弦，功率已在 `[0,1]` 内，进入扩散时映射到 `[-1,1]`。这些人工输入仅用于检查程序，不用于评估风电预测质量。

每次运行默认生成独立的 `outputs/synthetic/<时间戳>/<模型>/`：

- `checkpoint.pt`：模型、EMA、优化器、AMP、标准化统计量、数据顺序／位置及随机数状态。
- `scenarios.pt`：`[2, M, 24, 1]` 功率轨迹。
- `report.json`：损失、成功／跳过更新次数、参数量、采样诊断、耗时与显存。

恢复时指定检查点路径；`--updates` 表示追加的更新尝试数：

```bash
python train.py --device cuda --resume outputs/synthetic/<时间戳>/loop/checkpoint.pt --updates 2
```

恢复采用检查点中的模型、训练配置及合成数据配置；命令行的 `--amp`、`--seed`、微批次和累积配置不会覆盖它们。精确恢复要求相同 CPU／CUDA 设备类型。失败时的 `failure.pt` 是诊断快照，不能保证从部分累积步骤精确恢复。

## 本机已验证环境

2026-10-09 在 WSL2 的 `code_rl_clean` Conda 环境中通过实际 GPU 基础检查：Python 3.10.20、PyTorch 2.10.0+cu128、RTX 2060（6GB）、驱动 616.92。基础报告状态为 `CUDA_BASIC_OK`。模型代码使用 Python 3.10+ 和 PyTorch 2.10；独立环境检查脚本仍支持 Python 3.8+。

在本机 WSL 终端进入仓库后运行：

```bash
conda activate code_rl_clean
python check_wind_env.py
```

也可直接指定已验证的解释器，无需激活环境：

```bash
/home/u86177/miniconda3/envs/code_rl_clean/bin/python check_wind_env.py
```

默认 `base` 环境没有 PyTorch；`code_rl` 也安装了相同版本的 PyTorch，但本次实际 GPU 计算使用的是 `code_rl_clean`。无需重复安装依赖。受限执行沙箱可能无法访问 GPU，本次 GPU 验证在获准的沙箱外执行环境完成。

## 环境检查

使用实际准备训练的 Python 环境执行。脚本要求 Python 3.8+，报告生成仅依赖标准库；不安装软件、不下载数据。

```bash
# WSL / Linux
python3 check_wind_env.py
```

```powershell
# Windows PowerShell
python check_wind_env.py
```

报告写入当前目录下的 `outputs/environment/env_report_*.json`，包含系统、WSL、解释器、系统可见内存、磁盘、NVIDIA 驱动和 PyTorch 信息。PyTorch 检测使用独立子进程，默认 45 秒超时；存在可用 CUDA 时执行一次小型 FP32 前向、反向和 AdamW 更新，并核实参数发生变化。

| 状态 | 含义 |
| --- | --- |
| `CUDA_BASIC_OK` | 小型 GPU 计算通过；完整模型显存与 AMP 仍待验证 |
| `TORCH_NOT_INSTALLED` | 当前解释器缺少 PyTorch |
| `CUDA_UNAVAILABLE` | PyTorch 可导入，但 CUDA 不可用 |
| `NEEDS_REVIEW` | 导入、设备、超时或实际计算失败，查看报告错误字段 |

退出码 0 表示报告成功保存，不代表 GPU 验收通过。`nvidia-smi` 的错误独立保留，顶层状态来自 PyTorch 检查。可通过 `--device 1` 选择当前环境可见的另一块 GPU，通过 `--timeout 60` 调整检测超时。

报告只描述执行脚本的环境。WSL、Windows、远程会话和容器可能具有不同的 GPU 权限与 Python 软件包；不要仅凭一个受限会话的结果判断主机硬件故障。

## 文件与验证

- `check_wind_env.py`：可单独复制到训练电脑的检查脚本。
- `model.py`：三个模型的全部网络组件与初始化。
- `diffusion.py`：前向加噪、DDIM 与采样诊断。
- `train.py`：训练更新、合成数据、标准化、EMA 与断点。
- `tests/`：标准库 `unittest` 错误分支测试。
- `outputs/`：本机生成的报告与模型产物，不提交版本管理。

```bash
python -m unittest discover -s tests -v
python -m py_compile model.py diffusion.py train.py check_wind_env.py
```

Python 使用四空格缩进、`snake_case` 函数名。`pyproject.toml` 提供可选 Ruff 配置；已安装 Ruff 的开发环境可运行 `ruff check .` 和 `ruff format .`。运行模型不依赖 Ruff。

## 验收边界与后续工作

本次 27 项测试和 RTX 2060 上三个模型的 FP32／FP16 短训练、采样均通过，详见 [实测验收记录](docs/acceptance.md)。GPU AMP 断点续训也已通过。

测试覆盖参数量、共享梯度累加、循环早期梯度路径、同权重一轮等价、非因果注意力、多次更新后的上游梯度、断点重放及 DDIM 公式。CUDA 可用时还检查 AMP 溢出不更新 EMA；受限沙箱中该项自动跳过。

报告中的训练耗时含首次更新初始化开销，峰值显存是 PyTorch 的 allocated memory，包含当时存活的模型、EMA 和优化器，不是整卡占用。它们用于确认可运行，不能据此给三个架构作速度排名。少量更新后的场景不代表已学会条件分布。

后续才接入真实单风场数据、预测发布时间审计、按时间划分的训练／验证／测试，以及 CRPS、ES、VS 等概率评估。当前没有验证集最佳检查点选择，也没有正式对照实验结论。

CUDA 可用性接口依据 [PyTorch 官方文档](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.is_available.html)。
AMP 累积、反缩放与梯度裁剪顺序依据 [PyTorch 2.10 官方示例](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html)。
