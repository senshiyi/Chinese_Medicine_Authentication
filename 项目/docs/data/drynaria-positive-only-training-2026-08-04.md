# Drynaria_fortunei 仅正品训练与测试报告（2026-08-04）

## 实验目的与边界

本实验使用 `项目/References/raw/Drynaria_fortunei_authentic` 中的真实照片验证现有训练、锁定切分和评估流程。数据只包含正品，因此本报告不是正式鉴伪性能结论，模型也不能证明能够识别伪品或未知药材。

实验采用数据集标识 `Drynaria_fortunei` 作为声称药材和实际物种训练类别，标签依据为数据提供者于 2026-08-04 作出的确认；这不扩展为独立药典、显微、理化或分子鉴定结论。

## 数据与质量检查

- 原始图片：92 张 RGB JPEG，全部为 1920×1080，均可解码，无完全重复 SHA-256。
- 物理样本：46 个；同编号 `01/02` 为同一样本的两个视角，不同编号为不同样本。
- 标签：92 行均为 `authentic`，声称药材和实际物种实验标签均为 `Drynaria_fortunei`。
- 编码：manifest 为 UTF-8-SIG；验证报告为 `valid`，无 error 或 warning。
- 防泄漏：项目感知哈希识别出的 `0002/0016`、`0014/0018`、`0015/0025` 三组近似样本分别共享 `group_id`。

锁定切分使用 seed 42 和 0.70/0.15/0.15 比例：

| 分区 | 图片 | 物理样本 | group |
|---|---:|---:|---:|
| train | 64 | 32 | 30 |
| validation | 16 | 8 | 7 |
| test | 12 | 6 | 6 |

test 样本为 `df-0007`、`df-0011`、`df-0013`、`df-0020`、`df-0033`、`df-0039`。

## 训练配置与结果

- 模型：本地缓存并校验的 torchvision EfficientNet-B0 预训练骨干。
- 输入：224×224；batch size 8；CPU；混合精度关闭。
- 训练策略：冻结骨干，只训练三任务头；最多 10 轮，early-stop patience 3。
- 实际结果：完成 4 轮后早停；最佳 epoch 为 0；全局训练 step 为 32；最佳 selection score 为 0.75。
- checkpoint：schema v1，`provenance=real`，包含 manifest、完整 locked split 与训练配置指纹；未校准。
- checkpoint 大小：32,742,872 字节（31.23 MiB）。

## Locked test 结果

| 指标 | 图片级 | 样本级 |
|---|---:|---:|
| claimed macro F1 | 1.0 | 1.0 |
| species macro F1 | 1.0 | 1.0 |
| authenticity macro F1 | 0.5 | 0.5 |
| selection score | 0.75 | 0.75 |

- 选择性评估：eligible 12、accepted 12、coverage 1.0、selective accuracy 1.0、reject rate 0.0。
- group bootstrap selection score：下限/中位数/上限均为 0.75（200 次，seed 42）。
- `official_metrics=false`；`is_calibrated=false`；unknown test 状态为 `not_available`。

`claimed` 和 `species` 只有一个类别，因此 F1=1.0 是单类任务的必然结果之一，不能说明类别区分能力。真伪宏 F1=0.5 是因为指标固定评估正品/伪品两个类别，而数据中没有任何伪品。选择性准确率 1.0 只表示 12 张正品测试图在当前默认阈值下均被接受和判对，不能外推到伪品。

## 可复现信息

- manifest SHA-256：`4106522EC4BDE52C8FF0857A82EAEA5CDC5611878042BC5DA622B78E5052FCD9`
- locked split SHA-256：`AE19B5A117C9C39395F0BF9CA4F283856D90F40B6DE7FBE0BF406EF627AFF9AE`
- checkpoint SHA-256：`05E93BDB33F4108974C8378F6E072C718BB1394B68B30707E014BBA8DC17CEEE`
- 旧模型 `best_model.pth` 与 `epoch_001.pth` 均保持：`D67FB4D6268E386D191AE3374F4C5294F8A9854B916301D7BBA5DDE9B64F2606`

## 后续建议

在尝试正式鉴伪前，需要至少补充来源独立的伪品数据、unknown calibration 数据和从未参与训练/阈值选择的 unknown test 数据。还应补齐采购批次、来源、拍摄设备和鉴定依据，以防模型学习背景、设备或批次捷径。当前 checkpoint 保持未校准状态，不用于对外鉴定或正式准确率声明。
