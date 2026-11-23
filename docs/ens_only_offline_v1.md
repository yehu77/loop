# Berry Burn 日前 ENS-only 离线数据协议 v1

**验收结论。** 三个预定起报日中，2019-08-24 有一个完整的离线样本。另两个日期都有完整的官方功率目标，天气请求在身份认证处停止。本轮没有划分数据集。旧 HRES+ENS 审计继续保留自己的拒收结论。源提交为 `d387ede`。[冻结声明](../data_audit/ens_only_v1/protocol.json) · [分项结果与目标行索引](../data_audit/ens_only_v1/acceptance.json)

## 两个业务时钟

假定研究对象拥有业务 ENS 接入，当日所需的 ECMWF 00 UTC 产品在 12 UTC 发布预测前已正常交付。ECMWF 在 2017 年报告中记载，当时 00 UTC 分析的 15 天 ENS 在 08 UTC 可用；2019 年报 PDF 第 9、38 页记录日集合产品在 9 月又提前了 20 分钟。该调整晚于 8 月示例，未用于推算示例日到达时间。两份历史材料支持这一离线设定。样本的实际到达时间为 JSON null，`operational_availability_verified=false`。TIGGE 公共接入有起报后 48 小时延迟，不能用于这个发布后仅 12 小时就进入交付日的实时流程。[2017 年报告，第 21 页](https://www.ecmwf.int/sites/default/files/elibrary/2017/17281-how-ecmwf-has-addressed-requests-data-users.pdf) · [2019 年报](https://www.ecmwf.int/sites/default/files/elibrary/2020/19546-annual-report-2019.pdf) · [TIGGE 许可](https://ecds.ecmwf.int/licences/tigge-licence)

| UTC | 距初始化 | 距预测发布 | 角色 |
|---|---:|---:|---|
| 08-24 00:00 | 0 h | −12 h | ENS 初始化 |
| 08-24 12:00 | 12 h | 0 h | 离线协议的预测发布时间 |
| 08-25 00:00、06:00、12:00、18:00 | 24、30、36、42 h | 12、18、24、30 h | 目标日内四个原生天气节点 |
| 08-26 00:00 | 48 h | 36 h | 天气末端锚点，亦为最后一小时目标的右端点 |

24 个功率标签是 08-25 UTC 的左闭右开小时区间，起点分别距发布 12 至 35 h。每小时取 B1610 两个半小时的计量电量之和除以 1 h，保留 MW、真实零值、负值和来源行。每个引用指向保留结算版本的原始响应。这是实际送出功率；不使用估计可发容量或未来 BAV 过滤，无近期功率输入。作者成品表和 BAV 留作后续复核。[原始计量与请求清单](../data_audit/evidence/b1610)

## 位置和气象信息

站址代表点采用原项目 REPD **4261** 的 **57.446611°N、−3.475393°E**。核对英国官方 2026 Q3 CSV，该行是 Berry Burn、Operational、66.7 MW，英国国家格网为 E311550/N840550。使用 pyproj 3.7.2 的 `Transformer.from_crs(27700,4326,always_xy=True).transform(311550,840550)` 独立复算得到经纬度 (−3.475392537, 57.446611033)，按六位小数与协议一致。源文件 SHA256 为 `48e76c49f33a1090e78108d1ea9bfe689b35963ea3be1bb4f6516d038a5ff862`。旧审计引用的论文位置 **57.48712、−3.54822** 位于经线 −3.5° 另一侧。[官方 REPD CSV](https://assets.publishing.service.gov.uk/media/6af42ff0a4004bdbe54e54a5/REPD_Publication_Q3_2026.csv) · [论文附表](https://api.repository.cam.ac.uk/server/api/core/bitstreams/46bff039-dcbc-408d-aa14-0342e41a868d/content)

本协议直接使用已归档的附近四格点，纬度 57、57.5，经度 −4、−3.5。站址在其东界以东 0.024607°。没有空间插值，也没有把 6 小时曲线扩展成新的小时天气观测。每个起报需有 50 个可追踪成员、五个原生时刻和四个格点上的有限 u10/v10，量纲 m/s。若后续采用包围式插值，应重新取 −3.5、−3.0 两条经线。[首日 ENS 数据](../data_audit/evidence/ens.json)

## 实际取数和复现

探针按首日起报后的两个连续自然日顺序执行，选择发生在查看功率之前。每个新日期向 ECDS 提交一次限定日期、区域、变量、成员和步长的匿名请求，并向 Elexon 请求一次次日计量。

| 起报日 | 目标日 | 完整小时 | 原生 ENS | 离线样本 |
|---|---|---:|---|---|
| 2019-08-24 | 08-25 | 24 | 归档实读，1,000 个成员–时刻–格点组合 | 通过 |
| 2019-08-25 | 08-26 | 24 | HTTP 401，0 个取得 | 拒收 |
| 2019-08-26 | 08-27 | 24 | HTTP 401，0 个取得 | 拒收 |

两次 POST 请求体共 500 字节，四次探针响应体共 24,992 字节，其中天气错误响应共 484 字节。首日直接复用，无新增 Zenodo 下载。401 发生在认证阶段，两日 ENS 的实际档案覆盖仍未知。[逐请求参数、HTTP 状态、字节数与哈希](../data_audit/ens_only_v1/probes/ledger.json)

ECDS 提供可按日期和空间范围检索 TIGGE 的程序接口，需要用户认证并接受许可。获得授权后可用下列代码提交与探针相同的 MARS 风格参数；先在首日比对成员编号、坐标、时效、单位及风值，再扩展到两天。每个日期请求 500 个字段（2 分量 × 5 时刻 × 50 成员）。本轮没有实测认证取数的成功率或吞吐量。[ECMWF 迁移与请求示例](https://confluence.ecmwf.int/spaces/DAC/pages/462894687/User+guide+for+migration+of+S2S+and+TIGGE+to+ECDS)

```python
import cdsapi
from data_audit.ens_only_v1.probe import request_spec

client = cdsapi.Client(url="https://ecds.ecmwf.int/api")  # 用户自行配置 ECDS 凭据
client.retrieve("tigge-forecasts", request_spec("2019-08-25"), "ens_20190825.grib")
```

在仓库根目录执行离线验收。重新发起匿名探针时使用一个新的输出目录，以保留已提交的快照。

```bash
python -m data_audit.ens_only_v1.accept > /tmp/ens-only-replay.json
cmp /tmp/ens-only-replay.json data_audit/ens_only_v1/acceptance.json
python -m unittest discover -s tests -p 'test_ens_only_offline.py' -v
python -m data_audit.ens_only_v1.probe /tmp/ens-only-probe-new
```

本轮离线重放与归档 JSON 逐字节一致。新增验收测试 7 项、旧 Berry Burn 冻结审计测试 8 项全部通过，跳过 0 项。

进入均值与离散程度对照前，需取得更多匹配的起报窗口，按时间审计缺测、冻结互不重叠的训练/验证/测试日期，并固定同一批成员、格点和变量的均值与离散程度计算口径。单日样本验证的是接线和时钟。全量有效数与各集合窗口数仍未知。

Contains BMRS data © Elexon Limited copyright and database right 2026. [Elexon terms](https://www.elexon.co.uk/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/)
