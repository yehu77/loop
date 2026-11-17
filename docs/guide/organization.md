# 仓库组织原则

[文档首页](../README.md) · [代码地图](code-map.md)

## 本次整理解决什么

整理起点为 `e8e7339`。旧 README 共 268 行，持续追加研究过程，出现了开头“多种子待开展”与
后文“三种子已完成”并存的问题；原有 `docs/` 有 674 个已跟踪文件，但缺少按研究问题组织的总入口。

新的组织方式是：

1. **README**：项目目的、当前状态、最短使用路径。
2. **docs/guide/**：环境、代码地图、复现与维护约定。
3. **docs/experiments/catalog.json**：实验导航的结构化来源，记录报告、配置、证据、本地路径与归档状态。
4. **docs/experiments/README.md**：由目录数据生成的可读实验索引。
5. **Makefile / tools/**：轻量常用命令和只读导航，不参与训练算法。

历史报告中的阶段进度不会被统一改写；它们记录的是当时的判断。当前状态集中放在导航层，
避免每完成一轮实验都继续往 README 后面追加一段。

## 为什么保留原源码和归档路径

当前只是小型 PyTorch 研究工程，核心源码数量有限。更重要的是，已有配置、检查点和运行链路
按源文件路径与 SHA256 绑定身份。直接把根目录源码搬到 `src/`，或把 `outputs/` 的旧运行目录
改成新的分类目录，会影响导入、旧命令和恢复检查。

因此本次建立清晰的职责和导航层次，保留旧源码、冻结配置、数据、检查点及证据的路径和内容。
AGENTS.md 按原有要求保留。后续包结构迁移应作为单独的工程重构，用对应原版本回放与恢复验收，
而不是更新旧哈希来掩盖变化。

已有检查点、失败诊断、小规模验收和重复元数据不在本次整理中删除。若以后需要释放磁盘空间，
应另列具体可删除清单，并核对它们是否还被复现链路引用。

## 参考了哪些开源实践

- [facebookresearch/DiT](https://github.com/facebookresearch/DiT)：README 按设置、采样、训练组织，模型与训练入口直观。借鉴清楚的使用入口；该仓库已归档，不据此引入新的依赖或照搬训练实现。
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/#directory-structure)：区分源码、数据、模型产物、报告，并提供便捷命令。借鉴产物职责分离；本仓库保留已绑定的物理路径，用索引呈现逻辑分类。
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template#project-structure)：区分配置、实验、测试、脚本及日志。借鉴职责划分和可发现性；本次不引入 Lightning、Hydra 或新的训练框架。

这些是目录与文档组织的参考，不是风电实验结果的依据；没有复制外部模型代码。

## 维护与核验

```bash
make catalog
make experiment ID=history-architecture
make check
```

`make check` 校验目录记录与生成索引是否同步，以及报告、配置和归档路径是否存在；
它不校验模型性能或本地检查点哈希。实验专属的来源校验仍由原脚本负责。
本次整理的实际验证另见 [repository_organization_review.json](repository_organization_review.json)。

本次实际检查：README 由 268 行整理为 66 行；新文档的 124 个本地链接有效；
`make test` 为 149 通过 / 8 跳过；`make smoke` 的三模型 CPU 小检查通过。
13 个最新冻结源码、3,169 个历史保护文件和 20 个校正器结果归档文件哈希保持不变。
未新增正式训练、GPU 实验或测试集评分。
