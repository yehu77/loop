# 概率评价口径

实现位于 `metrics.py`，独立于训练和数据来源。输入为同一风场、相同发布时间及有效时间对齐的场景 `x: [B,M,H,1]` 和实测功率 `y: [B,H,1]`。函数要求完整、有限的浮点输入，在原设备用 FP64 计算；不填补、裁剪或重新归一化数据。当前风电任务应使用容量归一化功率。

## CRPS 与 Energy Score

本实现将 M 条场景视为等权经验分布。逐小时 CRPS 为：

\[
\operatorname{CRPS}_h=\frac1M\sum_m|x_{mh}-y_h|
-\frac1{2M^2}\sum_{m,n}|x_{mh}-x_{nh}|.
\]

整条轨迹的 Energy Score 为：

\[
\operatorname{ES}=\frac1M\sum_m\|x_m-y\|_2
-\frac1{2M^2}\sum_{m,n}\|x_m-x_n\|_2.
\]

两者均越小越好。CRPS 返回 `[B,H]`，ES 返回 `[B]`；ES 不除以 H 或其平方根，不能当作逐小时 CRPS 的均值。公式采用 [Gneiting 与 Raftery（2007）](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) 的损失方向。

分母明确为 `M²`，包括对角为零的有序样本对；没有采用 `M(M-1)` 的有限样本校正。比较模型时保持 M 一致。单场景时退化为绝对误差／欧氏误差。CRPS 用排序减少内存；ES 按场景分块累加全部距离，不随机抽取配对。

## 时间维度 Variogram Score

\[
\operatorname{VS}_p=\sum_{i\ne j}
\left(|y_i-y_j|^p-\frac1M\sum_m|x_{mi}-x_{mj}|^p\right)^2.
\]

这里 i、j 是未来小时，默认 `p=0.5`、所有非对角权重为 1、按有序小时对求和，不再除以小时对数量。返回 `[B]`，要求至少两个小时。该权重与缩放是本项目明确的起始选择；若外部基线使用归一化权重，应统一后再比较。

定义参考 [Scheuerer 与 Hamill（2015）](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf)。VS 关注小时之间的差异，但共同的功率水平偏移可保持这些差异不变，因此应同时报告 CRPS 和 ES。

## 覆盖率与区间宽度

对名义覆盖率 `q`，使用场景的 `(1-q)/2` 和 `(1+q)/2` 分位数形成中心区间，分位数采用线性插值。覆盖判定包含两个端点，宽度为上界减下界；均返回 `[B,H]` 信息。默认报告 80% 和 90% 区间。

汇总覆盖率是所有窗口与小时的点覆盖率，取值 `[0,1]`，不是整条轨迹同时落入区间的概率。宽度也不是越小越好，必须与覆盖率一起解释。报告同时保留逐小时 CRPS、覆盖率和宽度。

## 使用

```python
from metrics import evaluate_scenarios

report = evaluate_scenarios(scenarios, targets)
```

已有对齐的张量文件时：

```bash
python metrics.py --scenarios outputs/example/scenarios.pt \
  --targets outputs/example/targets.pt \
  --output outputs/example/metrics.json
```

两个 `.pt` 文件各自只保存一个张量，不是检查点或字典。CLI 在 CPU 上评分，报告包含 B、M、H 和计算口径，拒绝覆盖现有 JSON。裸张量不能证明时间对齐，数据管线必须先按风场、发布时间和有效时间核对对应关系。

本轮未修改训练损失或检查点选择规则，也没有用旧实验场景伪造真实目标。当前单元测试覆盖手算例子、直接配对公式、样本重排／复制不变性、相关结构变化、单样本边界、非有限值及命令行输出保护。

## 本轮验证

开发基于断点安全分支提交 `0114e4f`，使用独立分支 `codex/probabilistic-metrics`；没有合并 PR #1 或更新 main。2026-10-12 在现有 `code_rl_clean` 环境（Python 3.10.20、PyTorch 2.10.0+cu128）执行以下命令：

```bash
python -m unittest discover -s tests -p test_metrics.py -v
python -m unittest discover -s tests -v
python -m py_compile metrics.py tests/test_metrics.py
```

新增 13 项评价测试全部通过；全套 43 项中 42 项通过，1 项 CUDA 专用测试因当前受限执行环境无法使用 CUDA 而跳过；语法检查通过。本轮仅验证 CPU 评价代码与现有 CPU 回归路径，不认证 GPU 性能、真实预测效果或正式实验的统计结论。
