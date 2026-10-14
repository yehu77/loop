# 真实数据接口验收（2026-10-14）

基于 `01d3cb8` 开发数据管线；未更改 `model.py`、`diffusion.py`、训练损失或 `metrics.py` 的评分公式。复用 `code_rl_clean` 环境（Python 3.10.20、PyTorch 2.10.0+cu128、NumPy 2.2.6）。本轮真实数据训练使用既有 RTX 2060 的 CUDA 环境、FP32。

## 数据与测试

- 已确认的训练完整率规则实际选择 Zone 3，训练／验证／测试保留 547／71／90 个窗口。见 [数据协议](data_protocol.md) 和其中的冻结清单。
- 全套 59 项测试：58 通过，1 项 CUDA 专用 AMP 测试在受限 CPU 执行环境跳过。本轮没有重新认证 AMP。
- 新增 16 项数据／真实管线测试，包括跨午夜、跨年、闰日、缺行、NA、0/1 功率、重复主键、排序不变性、仅训练集拟合标准化、数据篡改与数据身份绑定。
- 评价入口拒绝重复／缺失／多余 ID、时间／风场／单位／版本错位；重排场景后按 ID 连接仍得到一致逐窗口分数。测试各日目标不同，避免相同目标掩盖错位。
- 额外验证不推进训练噪声 RNG、批次 RNG 或数据位置；随后一次训练更新与不插入验证的恢复路径精确一致（CPU）。
- 真实 CPU CLI 2 次更新、GPU CLI 20 次更新以及 GPU 从 20 续训至 22 均通过。
- 已保存的真实场景通过独立 `evaluate.py` 入口重放，身份与全部概率分数精确一致。

验证命令：

```bash
python -m unittest discover -s tests -v
python -m py_compile data.py prepare_data.py evaluate.py train_real.py train.py
python train_real.py --data outputs/data_v1_20261014/data.pt --device cuda \
  --updates 20 --validation-limit 4 --scenarios 16 --sampling-steps 10 \
  --validate-every 10 --output outputs/real_base_interface_20261014
python train_real.py --data outputs/data_v1_20261014/data.pt --device cuda \
  --resume outputs/real_base_interface_20261014/checkpoint_last.pt --updates 2 \
  --output outputs/real_base_resume_20261014
```

## 短运行结果

仅使用有效发布时间 2013-07-01 至 2013-07-04 的 4 个验证日，16 条场景、10 步 DDIM、固定验证噪声 seed=2026。RAND seed=2027，仅从 547 条训练轨迹整条抽样。微批次 4、累积 8 次，20 次成功更新共暴露 640 条训练样本（可重复）。

| 成功更新 | raw CRPS | EMA CRPS | raw 边界比例 | EMA 边界比例 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.25984090 | 0.25984090 | 99.2839% | 99.2839% |
| 10 | 0.25928511 | 0.25983562 | 98.5026% | 99.2839% |
| 20 | 0.25886690 | 0.25982060 | 97.5260% | 99.2839% |

同一验证面板的 RAND CRPS 为 **0.24036054**。预先固定的 EMA CRPS 选择规则选中第 20 次更新；没有从 raw／EMA 中事后挑更好的权重。

这 20 次更新的 EMA 尚未优于 RAND，场景仍大量挤在边界，不能据此判定扩散或循环机制有效／无效。4 天验证、16 条场景和 10 步采样只是接口配置，不是全验证集开发成绩。测试期的 90 个有效窗口未参与评分。

尚未执行 1,000 次成功更新的学习性试验，也未比较 LOOP／UNTIED；后续先做完整验证面板的 BASE-2 有限预算诊断。

## 可检查的本机产物

- `outputs/data_v1_20261014/`：数据包、清单、全样本元信息、丢弃记录与一个完整原始记录追溯样本。
- `outputs/real_base_interface_20261014/`：运行配置、数据清单、last/best 检查点、原始／EMA 验证历史、RAND、场景、目标和逐窗口分数。
- `outputs/real_base_resume_20261014/`：第 20→22 次成功更新的续训产物。
- `outputs/real_base_replayed_scores_20261014/`：独立重放的身份校验评价结果。
- `outputs/real_cpu_interface_20261014/`：CPU 入口检查产物。

所有原始数据、张量与检查点均留在版本管理之外。历史合成 GPU 验收记录 `docs/acceptance.md` 未被重写。
