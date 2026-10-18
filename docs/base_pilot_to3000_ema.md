# BASE-2 续训至 3,000 步与 EMA 历史诊断

本轮从 [1,000 步 pilot](base_pilot_v1_s42.md) 的正常 `checkpoint_last.pt` 续训，追加 2,000 次成功更新。起点提交为 `dfa553e`；模型、扩散、损失、训练器和真实训练入口与 `1f2110a` 相同。本轮仅增加独立诊断脚本、测试及记录。旧 pilot 的产物、成绩和检查点选择规则保持不变。

## 预先固定的设置

Zone 3；数据指纹 `403ff1adc1df6b1bd6dd79b24f206f164daf3161f829bf7a232fddf90a9168a4`。继续采用 FP32、seed 42、微批次 4、累积 8、学习率 `1e-4`、EMA 衰减 `0.999`。验证仍为全部 71 个完整有效日，每条件 100 场景、50 步 DDIM、chunk size 16、噪声 seed 2026，每 250 步评价 raw 和原 EMA。检查点仍按最低原 EMA 验证 CRPS 选择，raw 与重构 EMA 不参与选择。测试集不评分。

```bash
conda activate code_rl_clean
python train_real.py \
  --data outputs/data_v1_20261014/data.pt --device cuda \
  --resume outputs/real_base_pilot_v1_s42/checkpoint_last.pt \
  --updates 2000 --output outputs/real_base_pilot_v1_s42_to3000
```

原目录中的 last/best 均保留。续训配置读取检查点而不是新的 CLI 默认值；启动后已核对完整验证面板和采样预算。总计 3,000 次更新相当于 96,000 次训练窗口暴露，不代表新增独立样本。

## 离线诊断方法与验证

固定只比较第 3,000 步的 raw、原 EMA 和假设在第 1,000 步由 raw 重新起算的 EMA。重构逐个 `named_parameter` 应用：

```text
restarted_ema_3000 = original_ema_3000
                    + 0.999**2000 * (raw_1000 - original_ema_1000)
```

`diagnose_ema.py` 不创建优化器、不更新 Trainer、不保存训练检查点、不更新 best。它在 CPU 上以 FP64 计算校正再转回原 FP32；buffers 直接沿用终点原 EMA，符合现有 Trainer 在成功更新后复制 buffers 的规则。该恒等式在精确算术下成立；由于原检查点已经经历 FP32 舍入，实际结果不能宣称与假想逐步 FP32 重启 EMA 逐位相同。没有实际保存该假想逐步序列，不能给出真实模型重构误差的实测上界。

新增测试以固定随机参数序列显式递推至 3,000 步，对照公式重构；FP64 最大误差实测 `5.83e-16`（容差 `1e-12`），FP32 最大误差 `3.13e-7`（容差 `1e-5`）。这些是测试序列的误差，不是实际模型误差上界。另检查原状态不被修改、整数／浮点 buffers 原样保留、无效张量、不同数据／配置、故障标记、错误更新数与不完整续训证据的拒绝路径。区间指数使用成功更新数，跳步不计入。

仅有相同模型与数据配置不足以证明续训来源。本次在启动前记录起点 last/best、运行配置、数据包和训练源码 SHA-256；运行成功退出后才记录终点哈希。诊断同时核对恢复路径、逐次成功更新历史及续训起点 raw／EMA 评价与旧运行精确一致。这是可信本机运行记录的核验，不是对每一步优化器转移的密码学证明；不能为无关检查点事后伪造这份记录。

```bash
python -m unittest discover -s tests -v
python diagnose_ema.py \
  --start outputs/real_base_pilot_v1_s42/checkpoint_last.pt \
  --end outputs/real_base_pilot_v1_s42_to3000/checkpoint_last.pt \
  --data outputs/data_v1_20261014/data.pt \
  --lineage outputs/real_base_pilot_v1_s42_to3000.lineage.json \
  --device cuda --output outputs/real_base_ema_history_diagnostic_3000
```

三种权重分别重新创建相同验证噪声生成器，复用已有采样与身份对齐评价入口。诊断输出与主运行分开，执行前后验证四个 last/best 文件哈希未改变。

## 解释边界

这是单种子、单风场、单重启点的开发诊断，没有搜索多个重启点、衰减率或终点。参数平均不是预测平均；重构副本的改善只能支持这条参数轨迹上早期 EMA 历史的影响，不能证明唯一原因，也不能自动决定其他架构的配方。71 个验证日排除了 21/92 个缺失候选日；来源和 NWP 实际可用时间的未核验标记继续保留。下一版训练／权重选择规则应另行冻结，并对 BASE、LOOP、UNTIED 一致使用，不改写旧结果。
