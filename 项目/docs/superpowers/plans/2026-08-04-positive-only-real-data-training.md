# 仅正品真实数据训练与测试实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 `Drynaria_fortunei_authentic` 的 92 张真实照片完成一次样本级防泄漏训练和 locked test 测试，产出独立 checkpoint 与明确标注为非正式的评估报告。

**Architecture:** 复用现有三任务训练入口，不覆盖旧模型。46 个物理样本分别拥有 `01/02` 两个视角；同一 `sample_id` 固定在同一分区，感知哈希接近的跨编号样本使用共同 `group_id`，避免近似图片跨分区。

**Tech Stack:** Python 3.12、PyTorch 2.6、torchvision EfficientNet-B0、CSV UTF-8-SIG、YAML、pytest。

## Global Constraints

- 这是一项仅含正品的流程验证，不得宣称具备伪品识别能力或正式准确率。
- 用户确认 `0001`–`0046` 是不同物理样本，每个编号的 `01/02` 是同一样本的两个视角，全部为正品。
- 实验中使用数据集标识 `Drynaria_fortunei` 作为声称药材和实际物种类别；其依据记录为用户确认，不扩展为独立药典或分子鉴定结论。
- 原始照片只读，不重命名、不移动、不纳入 Git。
- 输出使用新实验目录，不覆盖 `best_model.pth`、`epoch_001.pth` 或正式多任务模型。

---

### Task 1: 建立实验清单和配置

**Files:**
- Create: `项目/Data/experiments/drynaria_positive_only/metadata.csv`
- Create: `项目/Files/config_positive_only.yaml`

**Interfaces:**
- Consumes: `项目/References/raw/Drynaria_fortunei_authentic/*.jpg`
- Produces: 92 行 known manifest 与独立训练配置。

- [x] **Step 1: 生成确定性 UTF-8-SIG 清单**

从文件名提取四位 `sample_id`；同编号两个视角复用该 ID。项目感知哈希检测到的 `0002/0016`、`0014/0018`、`0015/0025` 分别使用共同 `group_id`，其余样本以自身编号作为组。

- [x] **Step 2: 写入明确标签**

每行写入 `claimed_herb_name=Drynaria_fortunei`、`authenticity=authentic`、`actual_species=Drynaria_fortunei`、`species_annotation_status=verified`，并在 `notes` 记录用户确认日期和单类实验限制。

- [x] **Step 3: 创建 CPU 实验配置**

使用本地缓存的 EfficientNet-B0 预训练权重、224 输入、batch size 8、最多 10 epochs、early stop 3；设置 `unknown_manifest_path: null`、`provenance: real` 和独立模型/日志/评估路径。

### Task 2: 验证数据和锁定切分

**Files:**
- Create: `项目/Data/experiments/drynaria_positive_only/train.csv`
- Create: `项目/Data/experiments/drynaria_positive_only/val.csv`
- Create: `项目/Data/experiments/drynaria_positive_only/test.csv`
- Create: `项目/Data/experiments/drynaria_positive_only/splits.json`

**Interfaces:**
- Consumes: Task 1 manifest/config。
- Produces: 可复现、样本级防泄漏的 locked split。

- [x] **Step 1: 运行 manifest 验证**

验证 92 张图片均可解码、无 SHA 重复、标签和样本关系一致。

- [x] **Step 2: 生成固定切分**

使用 seed 42 和比例 0.70/0.15/0.15；确认每个 `sample_id` 只属于一个分区，且三个分区均非空。

- [x] **Step 3: 复核近似图防泄漏**

确认感知哈希接近的跨编号样本因为共享 `group_id` 而位于同一分区。

### Task 3: 训练实验模型

**Files:**
- Create: `项目/Models/experiments/drynaria_positive_only.pth`
- Create: `项目/Results/experiments/drynaria_positive_only/logs/validation_report.json`

**Interfaces:**
- Consumes: locked split、本地 EfficientNet-B0 权重。
- Produces: schema v1、带数据/切分/配置指纹的未校准 checkpoint。

- [x] **Step 1: 启动 CPU 训练**

Run: `.venv/python.exe 项目/Files/train.py --config 项目/Files/config_positive_only.yaml --device cpu`

- [x] **Step 2: 验证训练产物**

检查 checkpoint 存在、训练轮数非零、元数据 `provenance=real`，且旧模型哈希未变化。

### Task 4: locked test 测试与报告

**Files:**
- Create: `项目/Results/experiments/drynaria_positive_only/evaluation/evaluation.json`
- Create: `项目/Results/experiments/drynaria_positive_only/evaluation/evaluation.csv`

**Interfaces:**
- Consumes: Task 3 checkpoint 与 Task 2 test split。
- Produces: 已知正品测试结果和发布门禁结论。

- [x] **Step 1: 执行评估**

Run: `.venv/python.exe 项目/Files/evaluate.py --config 项目/Files/config_positive_only.yaml --json`

- [x] **Step 2: 检查报告语义**

报告必须显示 `official_metrics=false`，unknown test 为不可用；只报告这批正品上的召回/拟合表现，不称为鉴伪准确率。

- [x] **Step 3: 最终复核**

运行 `git diff --check`，确认原图未被跟踪、旧模型哈希未变化，并记录后续必须补充伪品和 unknown calibration/test 数据。
