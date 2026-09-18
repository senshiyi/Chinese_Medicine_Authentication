# 骨碎补闭集正伪鉴别项目完成设计

**日期：** 2026-08-09
**状态：** Task 9A component-constrained v2 设计已批准
**适用项目：** `Chinese_Medicine_Authentication`

## 1. 目标

使用现有 335 张骨碎补正品照片和 92 张骨碎补伪品照片，完成可复现、可审计的闭集正伪二分类训练与测试流程。项目交付包括完整数据清单、锁定切分、以真伪任务为主的模型训练、闭集概率校准、一次性 locked test 评估、候选模型安全晋升，以及供未来 Web 上传接口调用的结构化推理边界。

本阶段不宣称具备未知药材拒识能力，也不宣称能够识别伪品的实际物种。

## 2. 已确认的数据事实

- 正品目录：`项目/References/raw/Drynaria_fortunei_authentic`，335 张 JPEG。
- 伪品目录：`项目/References/raw/Drynaria_fortunei_fake`，92 张 JPEG。
- 共 427 张图片，全部为可解码的 RGB JPEG，尺寸均为 1920×1080。
- 当前审计未发现完全重复的 SHA-256 文件。
- 每张图片代表一个独立物理样本，不存在多视角样本关系。
- 所有样本声称的药材统一为 `Drynaria_fortunei`。
- 正品实际物种确认为 `Drynaria_fortunei`。
- 伪品实际物种未检验，不得根据文件夹、外观或模型预测回填物种标签。
- 数据没有采购批次、来源批次、拍摄设备或拍摄会话元数据。
- 当前没有 unknown calibration 或 unknown test 数据。

## 3. 项目边界

### 3.1 本阶段交付

- 闭集范围内的 `authentic`/`counterfeit` 二分类。
- 使用 validation 数据完成二分类概率温度缩放。
- 使用从未参与训练、模型选择或温度拟合的 locked test 数据进行一次性评估。
- 结构化实验结果、指标、配置快照和哈希审计。
- 候选模型通过门禁后原子覆盖正式模型。
- 保留未来 Web 上传页面可调用的 Path/bytes 推理服务和 JSON 响应边界。

### 3.2 明确不交付

- 未知药材或分布外样本的可靠拒识。
- 伪品实际物种识别。
- 正式开放集鉴定准确率。
- 当前版本的浏览器上传页面或 HTTP 服务。
- 基于批次、产地或设备的外部泛化结论。

## 4. 数据清单设计

重建完整 known manifest，共 427 行。每行字段遵循 `docs/data/manifest-v1.md`：

| 数据 | `sample_id` | `group_id` | `claimed_herb_name` | `authenticity` | `actual_species` | `species_annotation_status` |
|---|---|---|---|---|---|---|
| 正品 | 每图唯一 | 每图唯一 | `Drynaria_fortunei` | `authentic` | `Drynaria_fortunei` | `verified` |
| 伪品 | 每图唯一 | 每图唯一 | `Drynaria_fortunei` | `counterfeit` | 空 | `unverified` |

`sample_id` 使用带类别前缀的稳定编号，例如 `df-auth-0001` 和 `df-fake-0001`。`group_id` 与 `sample_id` 一一对应。生成器必须验证文件名连续性、目录标签、图片存在性和行数，并以稳定排序写出 UTF-8-SIG CSV。

由于缺少批次和拍摄会话信息，随机切分不能排除背景、设备或采集流程捷径。该限制必须写入所有正式报告。

## 5. 锁定切分

切分以 `(claimed_herb_name, authenticity)` 分层，使用 seed 42 和固定比例 0.70/0.15/0.15。按现有最大余数算法，预期分布为：

| 分区 | 正品 | 伪品 | 合计 |
|---|---:|---:|---:|
| train | 235 | 64 | 299 |
| validation | 50 | 14 | 64 |
| test | 50 | 14 | 64 |

`splits.json` 保存算法版本、seed、比例、完整 `group_id → partition` 映射和输入数据指纹。生成后禁止手工移动单张图片。manifest 或图片内容变化时必须创建新切分和新实验，旧 checkpoint 不得恢复训练。

### 5.1 版本化切分算法

V1: `group-stratified-v1` 行为冻结；默认配置与旧 `drynaria_positive_only` split 可重放。

V2: `group-stratified-near-duplicate-components-v2` 仅由显式配置启用。427 个 `sample_id`、427 个 `group_id` 保持一一对应；component 只作原子切分约束，⊥ 改写 provenance ID。

### 5.2 pHash 图与 component

V2 图节点=`group_id`。每图 pHash 固定 average-hash：RGB 解码→灰度→8×8 LANCZOS→像素 `>= mean` 记 1→16 位小写 hex。∀ 图对执行全量两两 Hamming；`distance <= perceptual_hash_distance`（inclusive）→边。component=无向图传递闭包；同 group 多图先归同节点，任意跨 group 图对成边即连接节点。阈值 4 正式固定，⊥ 降为 0、⊥ 近似索引漏边。

### 5.3 精确约束求解

每 component 计数向量=`(authentic_count, counterfeit_count)`。solver 使用稳定 component 排序、seed 42、精确 DP/回溯；! 同时满足：

- train=`(235,64)`；validation=`(50,14)`；test=`(50,14)`；
- component 不可拆分；
- 每个原始 `group_id` 恰好分配一次；
- 无精确解→fail closed，⊥ 贪心近似、⊥ 静默改计数。

一级可行后，二级目标：最小化各 partition multi-member component 标签分布对目标比例总绝对偏差。项=`abs(multi[p,y] * total_target[y] - target[p,y] * total_multi[y])`；总值=6 项和。

精确分解：mixed-label component 保留耦合 DP 状态；固定其分配后，authentic/counterfeit 配额、目标值、hash 和可加，按标签独立 weighted DP。两标签 group 集不相交，∴ 各子映射 sorted lex 最小→合并完整 mapping lex 最小。single-group multirow component 不计 multi-member 目标，但保持原子、进入精确配额 DP；singleton 仅作 exact filler。全局比较严格 `(objective, sum(sha256(seed, component_fingerprint, component_id, partition)), sorted(group_id → partition))`；⊥ quota state 提前按 hash 剪除不同 multi 分布。solver/objective/tie-break 版本写 audit。

### 5.4 独立泄漏 oracle

分配后使用独立全量两两实现重新计算所有图对 Hamming；任何 `distance <= threshold` 且 partition 不同→fail closed。oracle 不复用 component 候选索引或 solver 内部边集。正式 split 要求 `final_cross_partition_near_duplicate_count=0`。

### 5.5 V2 审计契约

`splits.json` 与 prepare report 必须写入且交叉一致：

- `algorithm_version`、`perceptual_hash_algorithm`、`perceptual_hash_distance`、inclusive 比较语义；
- `component_algorithm_version`、component/edge/singleton/mixed-label/multi-member counts、largest size、component fingerprint、完整 `group_id → component_id`、∀ component `(authentic,counterfeit)` 向量及其 fingerprint；
- `solver_algorithm_version`、`solver_status=optimal`、seed、目标函数名/值、tie-break 说明；
- 目标/实际 partition×label 二维计数；
- input fingerprint、mapping fingerprint、完整 `by_group`；
- independent oracle 版本、pair count、最终 cross count=0。

`apply_split` 按算法版本分派。V2 必须从当前 rows 重建图、component、solver 与所有 fingerprint/audit 字段；任一字段、seed、ratio、threshold、component、mapping、目标值或 cross count 篡改→拒绝。V1 继续走原重放路径。consumer ! `config.data.split_algorithm_version == assignment.algorithm_version`；V2 缺 `splits.json` → 在 checkpoint/report/logits 前拒绝并提示运行 `prepare_closed_set.py`；仅 V1 保留 missing-split fallback。train/evaluate config fingerprint ! 包含 split algorithm version 且实现一致。

### 5.6 原子 prepare

非 check-only：manifest、validation report、train/val/test CSV、splits JSON 全部先写同一 staging 目录并完成计数、重放、oracle、hash 校验；成功后才逐文件安全发布。发布前保留旧套件，发布异常回滚，禁止新 manifest+旧 split 混合。`--check-only` 全链只写临时目录，正式路径零变化。失败保留 stderr 诊断，正式套件零变化。

### 5.7 Task 9A 硬停点

V2 实现与正式 prepare 成功后停止。⊥ train、calibrate、evaluate、locked-test、predict、promote。真实 test 仅作为未读取指标的锁定 partition 工件写出；不得查看性能。

## 6. 模型与训练策略

继续使用共享视觉骨干与现有三任务输出结构，以保持 checkpoint 和未来接口的架构连续性：

- 声称药材头：保留接口兼容性；当前只有一个类别，不参与模型选择。
- 实际物种头：仅正品行产生监督；伪品行通过 mask 排除物种损失。
- 真伪头：本阶段主要任务，输出 `authentic`/`counterfeit` logits。

使用 torchvision EfficientNet-B0 的本地已校验预训练权重。第一阶段冻结骨干训练任务头；只有在 validation 真伪指标停滞且不存在明显过拟合时，才允许通过固定配置解冻最后若干层。训练使用类别权重或等价的平衡采样策略，具体实现只能启用一种，避免重复补偿类别不平衡。

最佳 checkpoint 的选择分数由真伪 Macro F1 和 Balanced Accuracy 构成，不再平均单类别 claimed/species 指标。test 数据不得参与早停、超参数选择、阈值选择或模型比较。

## 7. 闭集概率校准

本阶段新增与 open-set 校准严格分离的 `closed_set_temperature` 模式：

- 只使用 validation 的真伪 logits 拟合单个正温度参数。
- 优化目标为二分类负对数似然。
- 校准前后报告 NLL、Brier Score 和 ECE。
- 温度参数、validation 指纹、checkpoint 指纹和算法版本写入 checkpoint 元数据。
- test 只应用冻结后的温度参数，不参与温度拟合。
- 元数据必须同时保留 `open_set_calibrated=false`，不得把闭集温度缩放解释为未知样本拒识校准。

## 8. 评估设计

正式 locked test 报告至少包含：

- confusion matrix；
- Macro F1 和 Balanced Accuracy；
- 正品、伪品各自的 precision、recall、F1；
- AUROC 和 PR-AUC；
- Brier Score、ECE 和校准前后 NLL；
- 按样本 bootstrap 的 95% 置信区间；
- 图片数、类别数、数据来源状态和全部关键指纹。

候选模型晋升的预注册 validation 门槛为：

- Macro F1 ≥ 0.80；
- Balanced Accuracy ≥ 0.80；
- 伪品 recall ≥ 0.80；
- 所有用于晋升的指标为有限值；
- manifest、split、配置、校准和 checkpoint 指纹完整且一致。

locked test 只运行一次并如实报告。test 未达到同等门槛时，项目保留训练产物和失败报告，但不得标记为可发布模型。

## 9. 推理与用户文案

推理服务继续接受 `Path` 或图片 bytes，并返回可 JSON 序列化的结构化结果。结果必须包含：

- `experimental_closed_set=true`；
- `open_set_calibrated=false`；
- 校准后的正品/伪品概率；
- 模型、数据和切分版本；
- “仅适用于本数据集定义的骨碎补正品/伪品范围”警告。

由于没有 unknown 数据，正式 Web/API 鉴定模式保持关闭。实验模式不得把其他药材强制解释为正品或伪品。

正品实验结果使用“此药品为正品，有91.0%的可能该药材为正品。”的目标文案并附闭集概率和实验警告。伪品实际物种未检验，因此不得显示“实际应为某物种”；实际药材无法确定时，正式文案使用“实际药材无法确定，有93.0%的可能该药材为伪品，请人工复核。”；该概率是闭集模型概率，内部评估仍保留真伪预测和概率。

## 10. 模型覆盖与晋升

用户允许覆盖旧模型和原始模型，但训练过程不得直接写正式路径：

1. 训练写入同目录临时候选 checkpoint。
2. 完成 checkpoint 结构、指纹、validation 门槛、locked test 和推理回归检查。
3. 检查目标路径对应的加载器与 checkpoint schema 兼容。
4. 使用同卷原子替换晋升候选模型。
5. 记录覆盖前后的 SHA-256、时间、配置和运行 ID。

`best_model.pth`、`epoch_001.pth`、旧 `drynaria_positive_only.pth` 和正式多任务模型均可更新；不兼容新版 schema 的旧路径不得写入新版 checkpoint，除非相应加载入口同时升级并通过回归测试。

## 11. 错误处理

以下任一情况必须中止当前阶段，且不得覆盖正式模型：

- 图片无法解码、文件大小或像素超限；
- manifest 缺行、重复路径、标签非法或物种核验状态矛盾；
- sample/group 跨分区、完全重复或感知近重复跨分区；
- locked split 无法由 seed、比例和算法版本复现；
- 数据、split、配置或 checkpoint 指纹不匹配；
- 训练 loss、logit、概率或指标出现 NaN/Inf；
- validation 晋升指标未达到预注册门槛；
- 温度缩放失败或校准产物与 checkpoint 不兼容；
- 推理响应缺少实验/开放集状态标记；
- 临时候选 checkpoint 无法安全加载。

所有面向控制台的中文文本使用 UTF-8；Windows CP936/GBK 控制台通过现有编码初始化层兼容。

## 12. 测试策略

### 12.1 单元测试

- 427 行标签和唯一 sample/group 生成规则；
- 正品物种监督与伪品物种 mask；
- 类别权重只应用一次；
- 真伪主任务选择分数；
- Balanced Accuracy、分类别指标、AUROC、PR-AUC、Brier 和 ECE；
- 正温度参数拟合、序列化和指纹校验；
- 原子模型晋升和不兼容 schema 拒绝。

### 12.2 集成测试

- manifest → validation → locked split → train → closed-set calibration → evaluate 完整链路；
- test 数据不进入训练、早停或校准；
- 损坏图片、标签矛盾、指纹变化、校准缺失和指标不达标时失败关闭；
- 推理 Path/bytes 输入输出一致；
- 实验响应始终包含 closed/open-set 状态和警告；
- 覆盖后各模型入口能够加载对应 schema。

### 12.3 最终验收

- 427/427 图片可解码，335/92 标签数量正确；
- locked split 与预期 299/64/64 总数一致；
- 现有 210 项回归测试与新增测试全部通过；
- Ruff、`git diff --check` 和 CLI 编码检查通过；
- 完整训练、校准和 locked test 命令返回 0；
- 候选模型只有在全部晋升条件满足时才覆盖正式路径。

## 13. 产物与维护

每次运行保存独立 run 目录，至少包含：

- 原始配置快照和运行 ID；
- Python、PyTorch、torchvision 和操作系统版本；
- 随机种子及确定性设置；
- manifest、split、validation 和 test 指纹；
- 每轮训练历史、最佳轮次和早停原因；
- 校准参数和校准前后指标；
- JSON/CSV 评估报告和混淆矩阵；
- 候选/正式 checkpoint SHA-256 与晋升记录。

原始照片和模型二进制继续由 `.gitignore` 管理，不纳入 Git。可移植 manifest、`splits.json`、配置、规格、实施计划和实验总结纳入版本控制。

## 14. 后续升级条件

获得以下数据后再进入开放集版本：

- 与 known 数据来源独立的 unknown calibration 数据；
- 从未用于阈值选择的 unknown test 数据；
- 伪品实际物种的可追溯检验结果；
- 批次、来源、设备和拍摄会话元数据。

升级时保留本设计的闭集真伪任务作为一个独立、可回归的模型能力，不用开放集指标替代闭集指标。
