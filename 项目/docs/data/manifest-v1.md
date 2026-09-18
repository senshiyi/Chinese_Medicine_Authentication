# 数据清单规范 v1

本项目不从目录名猜测标签。训练、校准和测试数据必须通过 CSV 清单显式描述，内部统一按 UTF-8 处理；推荐保存为 UTF-8-SIG，历史 GBK 文件仅在自动识别成功时回退并产生警告。

## Known 数据

必填字段：

| 字段 | 含义 |
|---|---|
| `image_path` | 图片路径；相对路径不得越过配置的数据根目录 |
| `group_id` | 最高相关单元，如采购批次、来源批次或拍摄会话 |
| `sample_id` | 物理实体；同一药品的多个角度使用相同值 |
| `claimed_herb_name` | 包装、处方或送检信息所声称的药材 |
| `authenticity` | 只允许 `authentic` 或 `counterfeit` |
| `actual_species` | 专业鉴定的实际物种；多种物种用 `|` 分隔，未知时留空 |
| `species_annotation_status` | `verified`、`unverified` 或 `not_applicable` |

建议保留 `batch_id`、`origin`、`capture_device`、`collected_at`、`annotator`、`notes`，用于追溯偏差和复现实验。

示例：

```csv
image_path,group_id,sample_id,claimed_herb_name,authenticity,actual_species,species_annotation_status,batch_id,origin
known/g1/a-front.jpg,g1,a,当归,authentic,当归,verified,batch-001,甘肃
known/g1/a-side.jpg,g1,a,当归,authentic,当归,verified,batch-001,甘肃
known/g2/b.jpg,g2,b,当归,counterfeit,欧当归,verified,batch-002,未知
known/g3/c.jpg,g3,c,当归,counterfeit,,unverified,batch-003,未知
known/g4/d.jpg,g4,d,当归,counterfeit,当归|欧当归,verified,batch-004,未知
```

`actual_species` 未经可追溯专家、显微、理化或分子鉴定时必须标为 `unverified`，不得用模型预测回填。物种未知的行仍可训练声称药材和真伪任务，但不会产生物种监督。

## Unknown 数据

Unknown 数据必须独立于 known 数据采集，并显式标注 `dataset_role=calibration` 或 `dataset_role=test`。test 数据禁止用于选择阈值。

```csv
image_path,group_id,sample_id,unknown_type,dataset_role
unknown/u1.jpg,ug1,u1,unknown_herb,calibration
unknown/u2.jpg,ug2,u2,blurred,test
unknown/u3.jpg,ug3,u3,occluded,test
```

`unknown_type=unknown_herb` 的语义类别必须在训练 taxonomy 外；`blurred`、`occluded` 等质量异常可以来源于已知类别，但原始实体、批次和拍摄会话不能与 known 数据重叠。

## 切分与防泄漏

- 一个 `group_id` 只能属于 train、validation、test 中的一个分区。
- 同一 `sample_id` 的所有图片必须属于同一 `group_id`。
- 非空 `batch_id` 只能属于一个 `group_id`；同一 `sample_id` 的实际物种及核验状态必须一致。
- known 与 unknown calibration/test 之间不得共享 group、sample、完全重复或近重复图片。
- 同一来源、背景、包装、设备、批次或连续拍摄会话可能形成捷径时，应使用同一 `group_id`。
- 切分生成后必须锁定并保存 seed、比例、算法版本和数据指纹；数据变化后重新生成，不能手工挪动单张图片。
- 训练、校准和正式评估会核对完整 `splits.json`、validation CSV 与 unknown manifest 指纹，不能通过改角色列复用 test 数据。

## 图片要求

- 保留原始图；不要只保存社交软件压缩图或截图。
- 每个实体建议采集多个角度、尺度和光照，但避免所有真品与伪品使用不同背景或设备。
- 图片必须能被 Pillow 实际解码；扩展名或上传 MIME 不能代替内容验证。
- 排除重复、严重损坏、超大像素和来源不明文件。
- 最终性能结论只能来自来源独立、真实且未参与训练/校准的 locked test 数据。

## 最低规模建议

每个“声称药材 × 真伪”分层至少需要三个互不相关的 `group_id`，才能同时覆盖 train、validation、test。实际部署需要更多独立来源；样本数量不足时系统只验证流程，不应宣称准确率或正品鉴定能力。
