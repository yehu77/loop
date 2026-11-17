# 贡献说明

本项目的研究边界和断点安全规则见 [AGENTS.md](AGENTS.md)。当前实现范围与实验状态见
[代码地图](docs/guide/code-map.md)和[实验目录](docs/experiments/README.md)；早期验收文档保留原日期语境。

## 日常改动

使用独立分支。先找负责该行为的现有模块，保持四空格缩进和标准库 unittest 风格。
不要把模型、扩散、数据、指标重构夹带进实验结果归档。Ruff 配置在 `pyproject.toml`，
但不要对冻结文件做无关的全库格式化；源码字节变化也可能使旧断点拒绝恢复。

```bash
make check
make test
make smoke
```

按改动选择检查范围。仅修改导航时检查引用即可；训练／采样变化需要对应回归及实际 GPU 验收。
报告通过与跳过数量，并说明 CPU、CUDA、AMP、显存和吞吐中哪些确实测过。

## 新增实验

- 一个明确问题，一份冻结配置，一个独立输出目录；先确定主指标、样本面板、训练预算和停止条件。
- 当前新实验的报告用 `docs/<experiment_id>.md`，证据集中放在 `docs/experiments/<experiment_id>/`。
- 登记 `docs/experiments/catalog.json`，执行 `python tools/catalog.py list --markdown > docs/experiments/README.md`。
- 结果必须说明来源提交、数据／检查点身份、选择规则、逐日配对、参数与实际成本，以及解释边界。
- 测试集继续按既定规则封存；不要用它选择方法、风场或检查点。

`outputs/` 保留数据包、完整场景、检查点和大日志。Git 中优先提交紧凑的报告、配置、CSV、
摘要和哈希；避免复制多份完整样本元数据。已经冻结的归档保留不动。

## 提交与审阅

延续现有 Git 历史的简短动作式标题，例如 `Add ...`、`Fix ...`、`Archive ...`。
PR 说明具体问题与变化、起始提交、验证命令及环境、未覆盖边界；实验结果包含无收益或负向结果。
使用可审阅的独立分支，不强推、不自动合并，不重写主分支和历史实验结论。
