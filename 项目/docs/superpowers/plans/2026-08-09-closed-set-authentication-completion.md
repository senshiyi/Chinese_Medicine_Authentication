# Closed-Set Authentication Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 335 张骨碎补正品和 92 张骨碎补伪品照片建成可复现、可审计、可安全晋升的闭集正伪训练、校准、locked-test 评估和实验推理流程。

**Architecture:** 保留现有 EfficientNet 共享骨干和三任务 checkpoint schema，以真伪头作为本阶段唯一模型选择主任务；数据准备、训练、闭集温度校准、分区评估和模型晋升通过独立 CLI 串联。正式开放集推理继续失败关闭，新增显式的实验闭集推理响应供未来 Web 层调用，候选 checkpoint 只有在指纹、validation 门禁、一次性 locked test 和加载回归全部通过后才以同卷原子替换正式文件。

**Tech Stack:** Python 3.11/3.12、PyTorch 2.6.0、torchvision 0.21.0、scikit-learn 1.6.1、Pillow 11.1.0、PyYAML 6.0.2、pytest 8.3.4、Ruff 0.9.4；Windows PowerShell；控制台 UTF-8，CSV UTF-8-SIG，历史 GBK 只读兼容。

## Global Constraints

- 原始数据固定为 `项目/References/raw/Drynaria_fortunei_authentic` 的 335 张 JPEG 与 `项目/References/raw/Drynaria_fortunei_fake` 的 92 张 JPEG；每张图片是独立物理样本。
- 所有行 `claimed_herb_name=Drynaria_fortunei`；正品 `actual_species=Drynaria_fortunei,status=verified`；伪品 `actual_species` 为空且 `status=unverified`，禁止根据模型输出回填。
- 新实验数据写入 `项目/Data/experiments/drynaria_closed_set/`，不修改用户现有的 `drynaria_positive_only` 清单和切分。
- 固定分层切分参数为 seed 42、比例 0.70/0.15/0.15；预期 train/validation/test 为 299/64/64，其中各含正品 235/50/50、伪品 64/14/14。
- test 数据不得进入训练、早停、超参数选择、阈值选择或温度拟合；每个 run 的 locked test 最多执行一次。
- 真伪主任务选择分数固定为 `(authenticity_macro_f1 + authenticity_balanced_accuracy) / 2`；claimed/species 指标只报告、不参与选模。
- 类别不平衡只能使用真伪类别权重或 weighted sampler 之一；本配置固定使用类别权重并令 `data.use_weighted_sampler=false`。
- 闭集校准只拟合 validation 真伪 logits 的一个正温度；必须写明 `experimental_closed_set=true` 与 `open_set_calibrated=false`，不得宣称 unknown 拒识能力。
- 晋升 validation 门槛固定为 Macro F1 ≥ 0.80、Balanced Accuracy ≥ 0.80、counterfeit recall ≥ 0.80，且所有指标有限、全部关键指纹一致。
- locked test 低于同等门槛时保留候选与报告，但不得晋升正式模型；不得通过重复查看 test 调参。
- 训练始终写候选路径；模型覆盖仅由独立晋升命令执行，先安全加载、校验 schema 与哈希，再使用同目录临时文件和 `os.replace`。
- 正式开放集 Web/API 模式保持关闭；本版本不实现浏览器页面或 HTTP 服务，只提供 Path/bytes 与 JSON 原生类型的实验闭集边界。
- 若伪品实际物种未经检验，用户文案不得显示“实际应为某物种”；若实际药材无法确定，文案必须为“实际药材无法确定，有93.0%的可能该药材为伪品，请人工复核。”；该概率是闭集模型概率，结构化内部字段仍保留闭集概率供审计。
- 所有面向控制台的中文使用现有 `configure_utf8_stdio()`；源文件和 JSON 使用 UTF-8，CSV 使用 UTF-8-SIG，读取端保留 UTF-8-SIG/GBK 兼容。
- 原始图片、`.pth` 和 `项目/Results/` 不纳入 Git；配置、manifest、`splits.json`、规格、计划与实验总结纳入 Git。
- 不改动工作区中与本计划无关的用户文件；每次 `git add` 必须列出精确路径。

---

## File Structure Map

### New files

- `项目/Files/herb_authentication/data/preparation.py`：从两个受控原始目录生成稳定、原子写入的 427 行 known manifest。
- `项目/Files/prepare_closed_set.py`：数据准备和锁定切分 CLI；完成目录审计、manifest 校验、切分写出与计数验收。
- `项目/Files/config_closed_set.yaml`：本次闭集实验的唯一运行配置。
- `项目/Files/herb_authentication/training/run_artifacts.py`：run ID、环境快照、训练摘要和指纹写入。
- `项目/Files/herb_authentication/inference/closed_set.py`：闭集温度参数、实验响应 schema 和闭集文案。
- `项目/Files/herb_authentication/training/promotion.py`：晋升门禁、checkpoint 兼容检查、哈希审计和原子发布。
- `项目/Files/promote.py`：只负责候选模型晋升的 CLI。
- `tests/unit/test_data_preparation.py`：manifest 生成规则测试。
- `tests/integration/test_prepare_closed_set_cli.py`：数据准备至锁定切分的 CLI 集成测试。
- `tests/unit/test_authenticity_metrics.py`：真伪分类、概率质量和 bootstrap 指标测试。
- `tests/unit/test_closed_set_calibration.py`：单温度拟合、序列化和指纹测试。
- `tests/integration/test_closed_set_calibrate_cli.py`：无 unknown 数据的闭集校准 CLI 测试。
- `tests/integration/test_partition_evaluation.py`：validation/locked-test 隔离、报告与一次性账本测试。
- `tests/unit/test_closed_set_inference.py`：Path/bytes 一致性、状态标记、文案和 JSON 测试。
- `tests/unit/test_promotion.py`：晋升门禁、schema 拒绝、旧文件保全与原子替换测试。
- `tests/integration/test_closed_set_pipeline.py`：小型合成数据的 prepare→train→calibrate→evaluate→promote 冒烟链路。
- `docs/experiments/2026-08-09-drynaria-closed-set.md`：真实运行结果、限制、命令、哈希及问题记录。

### Modified files

- `项目/Files/herb_authentication/data/__init__.py`：导出 manifest 准备公共接口。
- `项目/Files/herb_authentication/config.py`：加入 run、闭集校准、评估分区和晋升门槛配置并做交叉校验。
- `项目/Files/herb_authentication/training/metrics.py`：加入 Balanced Accuracy，并把 selection score 改为真伪主任务组合。
- `项目/Files/herb_authentication/training/trainer.py`：接收且应用一次真伪类别权重，保存训练历史与主任务选模元数据。
- `项目/Files/train.py`：装配类别权重、独立 run 目录、候选输出与可复现环境快照。
- `项目/Files/herb_authentication/inference/calibration.py`：保留既有开放集校准并导出通用正温度拟合原语。
- `项目/Files/calibrate.py`：增加互斥的 `--mode closed-set-temperature` 路径，不要求 unknown CSV。
- `项目/Files/herb_authentication/evaluation.py`：增加完整二分类与校准指标、混淆矩阵及逐指标 bootstrap 区间。
- `项目/Files/evaluate.py`：显式选择 validation 或 locked-test，验证闭集校准指纹并维护 locked-test 账本。
- `项目/Files/herb_authentication/inference/service.py`：加载闭集校准元数据并提供 `predict_experimental_closed_set(Path|bytes)`。
- `项目/Files/herb_authentication/inference/__init__.py`：导出闭集类型。
- `项目/Files/predict.py`：加入显式 `--experimental-closed-set`，默认正式模式行为不变。
- `项目/Files/herb_authentication/training/__init__.py`：导出 run 与晋升接口。
- `tests/unit/test_config.py`、`tests/unit/test_metrics.py`、`tests/unit/test_trainer.py`、`tests/unit/test_trainer_selection.py`：更新配置、主指标和类别权重回归。
- `tests/integration/test_calibrate_cli.py`、`tests/integration/test_evaluate_cli.py`、`tests/integration/test_predict_cli.py`、`tests/integration/test_release_gates.py`：保持开放集兼容并覆盖新模式。
- `使用指南.md`：写入从数据准备到晋升的命令、实验边界、编码要求和未来 Web 调用接口。
- `.gitignore`：忽略 run 中的二进制/临时工件，但保留可移植审计 JSON/CSV 的现有策略。

---

### Task 1: Deterministic Full Manifest and Locked Split

**Files:**
- Create: `项目/Files/herb_authentication/data/preparation.py`
- Create: `项目/Files/prepare_closed_set.py`
- Modify: `项目/Files/herb_authentication/data/__init__.py`
- Test: `tests/unit/test_data_preparation.py`
- Test: `tests/integration/test_prepare_closed_set_cli.py`

**Interfaces:**
- Consumes: `validate_manifest_pair(...)`, `create_group_splits(rows, ratios, seed)`, `write_split_artifacts(rows, assignment, output_dir)`。
- Produces: `ManifestSource(directory: Path, filename_pattern: Pattern[str], id_prefix: str, authenticity: str, actual_species: str | None, species_annotation_status: str)`；`build_closed_set_manifest(sources: Sequence[ManifestSource], data_root: Path, output_path: Path, claimed_herb_name: str) -> ManifestBuildResult`；`ManifestBuildResult(path: Path, rows: int, counts: Mapping[str, int], sha256: str)`。

- [ ] **Step 1: Write failing manifest-generation tests**

```python
def write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(127, 64, 32)).save(path, format="JPEG")


def fixture_sources(authentic: Path, fake: Path) -> tuple[ManifestSource, ManifestSource]:
    return (
        ManifestSource(authentic, re.compile(r"Drynaria-fortunei_AUTHENTIC_(\d{4})\.jpg"), "df-auth", "authentic", "Drynaria_fortunei", "verified"),
        ManifestSource(fake, re.compile(r"Drynaria-fortunei_FAKE_(\d{4})\.jpg"), "df-fake", "counterfeit", None, "unverified"),
    )


def test_build_closed_set_manifest_assigns_one_sample_and_group_per_file(tmp_path: Path) -> None:
    authentic = tmp_path / "raw" / "Drynaria_fortunei_authentic"
    fake = tmp_path / "raw" / "Drynaria_fortunei_fake"
    write_jpeg(authentic / "Drynaria-fortunei_AUTHENTIC_0001.jpg")
    write_jpeg(authentic / "Drynaria-fortunei_AUTHENTIC_0002.jpg")
    write_jpeg(fake / "Drynaria-fortunei_FAKE_0001.jpg")

    result = build_closed_set_manifest(
        sources=fixture_sources(authentic, fake),
        data_root=tmp_path,
        output_path=tmp_path / "metadata.csv",
        claimed_herb_name="Drynaria_fortunei",
    )

    rows = read_csv_dicts(result.path, encoding="utf-8-sig").rows
    assert [row["sample_id"] for row in rows] == ["df-auth-0001", "df-auth-0002", "df-fake-0001"]
    assert [row["group_id"] for row in rows] == ["df-auth-0001", "df-auth-0002", "df-fake-0001"]
    assert rows[-1]["actual_species"] == ""
    assert rows[-1]["species_annotation_status"] == "unverified"
```

Also test: unexpected filename, missing numeric index, duplicate index, symlink/path escape, non-JPEG file, count mismatch, and failed write preserve an existing output file.

- [ ] **Step 2: Run the focused tests and verify red state**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_data_preparation.py tests/integration/test_prepare_closed_set_cli.py -q`

Expected: collection fails with `ModuleNotFoundError: herb_authentication.data.preparation` or missing script.

- [ ] **Step 3: Implement stable manifest generation**

```python
@dataclass(frozen=True, slots=True)
class ManifestBuildResult:
    path: Path
    rows: int
    counts: Mapping[str, int]
    sha256: str


def _row(source: ManifestSource, image: Path, index: int, data_root: Path, claimed: str) -> dict[str, str]:
    sample_id = f"{source.id_prefix}-{index:04d}"
    return {
        "image_path": image.resolve().relative_to(data_root.resolve()).as_posix(),
        "group_id": sample_id,
        "sample_id": sample_id,
        "claimed_herb_name": claimed,
        "authenticity": source.authenticity,
        "actual_species": source.actual_species or "",
        "species_annotation_status": source.species_annotation_status,
        "batch_id": "",
        "origin": "",
        "capture_device": "",
        "collected_at": "",
        "annotator": "",
        "notes": "independent physical sample; batch/source/device/session unavailable",
    }
```

Sort by `(authenticity order authentic/counterfeit, numeric index, filename)`; require indexes exactly `1..N`; write a same-directory temporary UTF-8-SIG CSV, read it back, then `os.replace`.

- [ ] **Step 4: Implement preparation CLI and real-data count gate**

`prepare_closed_set.py` must call `configure_utf8_stdio()`, build the manifest, validate all images, create/apply seed-42 split, write `train.csv`, `val.csv`, `test.csv`, `splits.json`, and reject unless counts equal:

```python
EXPECTED = {
    "all": {"authentic": 335, "counterfeit": 92, "total": 427},
    "train": {"authentic": 235, "counterfeit": 64, "total": 299},
    "val": {"authentic": 50, "counterfeit": 14, "total": 64},
    "test": {"authentic": 50, "counterfeit": 14, "total": 64},
}
```

CLI prints one UTF-8 JSON object containing output paths, counts, manifest SHA-256, split input fingerprint, and split-file SHA-256.

- [ ] **Step 5: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_data_preparation.py tests/integration/test_prepare_closed_set_cli.py -q`

Expected: all focused tests pass.

- [ ] **Step 6: Commit the data preparation feature**

```powershell
git add 项目/Files/herb_authentication/data/preparation.py 项目/Files/herb_authentication/data/__init__.py 项目/Files/prepare_closed_set.py tests/unit/test_data_preparation.py tests/integration/test_prepare_closed_set_cli.py
git commit -m "feat: prepare locked closed-set dataset"
```

---

### Task 2: Closed-Set Configuration and Run Provenance

**Files:**
- Create: `项目/Files/config_closed_set.yaml`
- Create: `项目/Files/herb_authentication/training/run_artifacts.py`
- Modify: `项目/Files/herb_authentication/config.py:17-328`
- Modify: `项目/Files/herb_authentication/training/__init__.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_run_artifacts.py`

**Interfaces:**
- Produces: `CalibrationConfig.mode: Literal["open_set", "closed_set_temperature"]`；`PromotionConfig` with exact three thresholds and target path；`RunContext(run_id: str, root: Path, config_snapshot: Path)`；`create_run_context(config: AppConfig, requested_run_id: str | None) -> RunContext`；`write_run_metadata(context, payload) -> Path`；`sha256_file(path) -> str`。
- Consumes: Task 1 manifest/split paths and existing `write_json_utf8`.

- [ ] **Step 1: Add failing strict-config tests**

```python
def write_config(tmp_path: Path, **sections: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(sections, allow_unicode=True), encoding="utf-8")
    return path


def test_closed_set_config_rejects_double_class_balancing(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        data={"use_weighted_sampler": True},
        training={"use_authenticity_class_weights": True},
        calibration={"mode": "closed_set_temperature"},
    )
    with pytest.raises(ConfigurationError, match="只能启用一种"):
        load_config(path)


def test_promotion_thresholds_are_probabilities(tmp_path: Path) -> None:
    path = write_config(tmp_path, promotion={"min_macro_f1": 1.1})
    with pytest.raises(ConfigurationError, match="promotion.min_macro_f1"):
        load_config(path)
```

- [ ] **Step 2: Run tests and verify red state**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/unit/test_run_artifacts.py -q`

Expected: failures for unknown `promotion`, missing `mode`, and missing run artifact module.

- [ ] **Step 3: Extend typed configuration without weakening unknown-key rejection**

```python
@dataclass(frozen=True, slots=True)
class PromotionConfig:
    min_macro_f1: float = 0.80
    min_balanced_accuracy: float = 0.80
    min_counterfeit_recall: float = 0.80
    target_checkpoint_path: Path = Path("项目/Models/best_multitask_model.calibrated.pth")


@dataclass(frozen=True, slots=True)
class RunConfig:
    root_dir: Path = Path("项目/Results/experiments/drynaria_closed_set/runs")
```

Add `TrainingConfig.use_authenticity_class_weights: bool = True`; add `CalibrationConfig.mode`, `CalibrationConfig.algorithm_version="closed-set-temperature-v1"`; add `run` and `promotion` to `AppConfig`, root keys, builders, and validation. Reject `mode=closed_set_temperature` if `data.unknown_manifest_path` is not `None` or if both balancing mechanisms are true.

Change the three task-loss validators from strictly positive to finite nonnegative values and reject only when all three are zero; this permits the one-class claimed head to retain schema compatibility without contributing a constant loss.

- [ ] **Step 4: Add immutable run directories and environment snapshots**

Use `YYYYMMDDTHHMMSSZ-` followed by the first eight hexadecimal characters of the config SHA-256 when no run ID is provided; reject an existing non-empty run directory. Snapshot the source YAML as UTF-8 and write `environment.json` with Python, torch, torchvision, OS, seed, deterministic settings, config SHA-256 and command arguments. Do not copy raw images or checkpoint tensors into JSON.

- [ ] **Step 5: Add the exact real configuration**

```yaml
project_root: "../.."
data:
  manifest_path: "项目/Data/experiments/drynaria_closed_set/metadata.csv"
  unknown_manifest_path: null
  data_root: "项目/References"
  provenance: "real"
  text_encoding: "utf-8-sig"
  image_size: 224
  batch_size: 8
  num_workers: 0
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
  max_image_bytes: 20971520
  max_image_pixels: 40000000
  perceptual_hash_distance: 4
  use_weighted_sampler: false
model:
  backbone: "efficientnet_b0"
  pretrained: true
  freeze_backbone: true
  unfreeze_epoch: 100
  dropout: 0.3
training:
  epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  claimed_herb_weight: 0.0
  actual_species_weight: 0.25
  authenticity_weight: 1.0
  label_smoothing: 0.0
  early_stop_patience: 5
  mixed_precision: false
  gradient_clip: 1.0
  min_species_supervision_coverage: 0.0
  use_authenticity_class_weights: true
  checkpoint_path: "项目/Models/experiments/drynaria_closed_set.candidate.pth"
  log_dir: "项目/Results/experiments/drynaria_closed_set/logs"
calibration:
  mode: "closed_set_temperature"
  algorithm_version: "closed-set-temperature-v1"
  max_unknown_accept_rate: 0.05
  min_known_selective_accuracy: 0.95
  output_checkpoint_path: "项目/Models/experiments/drynaria_closed_set.calibrated.candidate.pth"
evaluation:
  report_dir: "项目/Results/experiments/drynaria_closed_set/evaluation"
  bootstrap_samples: 2000
inference:
  checkpoint_path: "项目/Models/best_multitask_model.calibrated.pth"
  device: "cpu"
  allow_uncalibrated: false
  max_image_bytes: 20971520
  max_image_pixels: 40000000
run:
  root_dir: "项目/Results/experiments/drynaria_closed_set/runs"
promotion:
  min_macro_f1: 0.80
  min_balanced_accuracy: 0.80
  min_counterfeit_recall: 0.80
  target_checkpoint_path: "项目/Models/best_multitask_model.calibrated.pth"
```

- [ ] **Step 6: Verify config and run-artifact tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/unit/test_run_artifacts.py -q`

Expected: all focused tests pass; loading `config_closed_set.yaml` resolves all paths under this project and reports closed-set mode.

Run: `.\.venv\Scripts\python.exe 项目\Files\prepare_closed_set.py --config 项目\Files\config_closed_set.yaml --check-only`

Expected: exit 0; JSON reports 427, 335, 92, and 299/64/64; --check-only does not write files.

- [ ] **Step 7: Commit configuration and provenance**

```powershell
git add 项目/Files/config_closed_set.yaml 项目/Files/herb_authentication/config.py 项目/Files/herb_authentication/training/run_artifacts.py 项目/Files/herb_authentication/training/__init__.py tests/unit/test_config.py tests/unit/test_run_artifacts.py
git commit -m "feat: configure auditable closed-set runs"
```

---

### Task 3: Authenticity-First Training and Single Class Balancing

**Files:**
- Modify: `项目/Files/herb_authentication/training/metrics.py:30-135`
- Modify: `项目/Files/herb_authentication/training/trainer.py:41-350`
- Modify: `项目/Files/train.py:161-358`
- Test: `tests/unit/test_metrics.py`
- Test: `tests/unit/test_trainer.py`
- Test: `tests/unit/test_trainer_selection.py`
- Test: `tests/integration/test_train_assembly.py`

**Interfaces:**
- Produces: `EntityMetrics.authenticity_balanced_accuracy: float`; selection score formula fixed above；`compute_authenticity_class_weights(targets: Sequence[int], classes: int = 2) -> torch.Tensor`；`Trainer(..., authenticity_class_weight: torch.Tensor | None = None)`。
- Consumes: Task 2 `TrainingConfig.use_authenticity_class_weights` and run artifact helpers.

- [ ] **Step 1: Replace the old harmonic-selection expectation with failing primary-task tests**

```python
def test_selection_score_uses_only_authenticity_macro_f1_and_balanced_accuracy() -> None:
    predictions = aggregate_by_sample(make_predictions())
    predictions.claimed_probabilities[:] = predictions.claimed_probabilities.flip(1)
    predictions.species_probabilities[:] = predictions.species_probabilities.flip(1)
    metrics = compute_entity_metrics(predictions, min_species_supervision_coverage=0.0)
    assert metrics.authenticity_macro_f1 == 1.0
    assert metrics.authenticity_balanced_accuracy == 1.0
    assert metrics.selection_score == 1.0
    assert metrics.selection_tasks == ("authenticity_macro_f1", "authenticity_balanced_accuracy")


def test_authenticity_class_weights_are_inverse_frequency_and_finite() -> None:
    weights = compute_authenticity_class_weights([0, 0, 0, 1])
    assert weights.tolist() == pytest.approx([2 / 3, 2.0])
```

- [ ] **Step 2: Run focused tests and verify failures describe the old behavior**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_metrics.py tests/unit/test_trainer_selection.py tests/unit/test_trainer.py -q`

Expected: failures mention missing `authenticity_balanced_accuracy`, old `selection_tasks`, or missing class-weight helper.

- [ ] **Step 3: Implement balanced accuracy and new selection score**

Compute each class recall with a zero-denominator guard, require both authenticity classes in validation, and use:

```python
selection_score = (authenticity_macro_f1 + authenticity_balanced_accuracy) / 2.0
selection_tasks = ("authenticity_macro_f1", "authenticity_balanced_accuracy")
```

Continue reporting claimed/species Macro F1 and coverage unchanged, so checkpoint consumers retain diagnostic fields.

- [ ] **Step 4: Thread exactly one class-balance mechanism through training**

Store `self.authenticity_class_weight` on the resolved device and pass it only here:

```python
losses = compute_multitask_loss(
    logits,
    targets,
    masks,
    self.loss_weights,
    label_smoothing=self.label_smoothing,
    authenticity_class_weight=self.authenticity_class_weight,
)
```

In `train.py`, derive targets from `train_rows` using `labels.authenticity_classes`, reject a missing class, and only construct weights when `use_authenticity_class_weights=true`. Do not create a weighted sampler.

- [ ] **Step 5: Persist training history and selection contract**

Write `training_history.json` under the run directory with every epoch loss, true diagnostic metrics, best epoch, early-stop reason, selection formula string, class counts, and applied class weights. Add the run ID, selection formula, and `counterfeit_species_verified=false` (derived by requiring every counterfeit training row to have verified species; false here) to checkpoint metadata. Every numeric value must be checked with `math.isfinite` before serialization.

- [ ] **Step 6: Run training-focused regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_metrics.py tests/unit/test_trainer.py tests/unit/test_trainer_selection.py tests/integration/test_train_assembly.py tests/integration/test_resume_training.py -q`

Expected: all focused tests pass; a validation set missing either authenticity class fails closed before checkpoint selection.

- [ ] **Step 7: Commit authenticity-first training**

```powershell
git add 项目/Files/herb_authentication/training/metrics.py 项目/Files/herb_authentication/training/trainer.py 项目/Files/train.py tests/unit/test_metrics.py tests/unit/test_trainer.py tests/unit/test_trainer_selection.py tests/integration/test_train_assembly.py
git commit -m "feat: select checkpoints by authenticity metrics"
```

---

### Task 4: Closed-Set Authenticity Temperature Calibration

**Files:**
- Create: `项目/Files/herb_authentication/inference/closed_set.py`
- Modify: `项目/Files/herb_authentication/inference/calibration.py:252-370`
- Modify: `项目/Files/herb_authentication/inference/__init__.py`
- Modify: `项目/Files/calibrate.py:25-146`
- Test: `tests/unit/test_closed_set_calibration.py`
- Test: `tests/integration/test_closed_set_calibrate_cli.py`
- Modify test: `tests/integration/test_calibrate_cli.py`

**Interfaces:**
- Produces: `ClosedSetCalibration(temperature: float, algorithm_version: str, validation_fingerprint: str, checkpoint_fingerprint: str, nll_before: float, nll_after: float, brier_before: float, brier_after: float, ece_before: float, ece_after: float)` with strict `to_dict/from_dict`; `fit_closed_set_temperature(logits: Tensor, targets: Tensor, *, validation_fingerprint: str, checkpoint_fingerprint: str, algorithm_version: str) -> ClosedSetCalibration`。
- Consumes: existing `_fit_temperature` behavior refactored into public `fit_positive_temperature(logits, targets, loss_kind="cross_entropy") -> float`; Task 2 calibration mode and fingerprints.

- [ ] **Step 1: Write failing calibration tests**

```python
def test_closed_set_temperature_is_positive_deterministic_and_improves_nll() -> None:
    logits = torch.tensor([[8.0, 0.0], [7.0, 0.0], [0.0, 8.0], [0.0, 7.0]])
    targets = torch.tensor([0, 1, 1, 0])
    first = fit_closed_set_temperature(
        logits, targets,
        validation_fingerprint="a" * 64,
        checkpoint_fingerprint="b" * 64,
        algorithm_version="closed-set-temperature-v1",
    )
    second = fit_closed_set_temperature(logits, targets, validation_fingerprint="a" * 64,
        checkpoint_fingerprint="b" * 64, algorithm_version="closed-set-temperature-v1")
    assert first == second
    assert first.temperature > 0
    assert first.nll_after <= first.nll_before + 1e-9
    assert ClosedSetCalibration.from_dict(first.to_dict()) == first
```

Also reject single-class targets, NaN/Inf, wrong shape, non-64-character fingerprints, and a calibration block whose recorded checkpoint hash differs from the loaded candidate.

- [ ] **Step 2: Run tests and verify red state**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_closed_set_calibration.py tests/integration/test_closed_set_calibrate_cli.py -q`

Expected: imports or new CLI mode fail.

- [ ] **Step 3: Implement isolated closed-set calibration**

Fit `log_temperature` with LBFGS, use `temperature=exp(log_temperature).clamp(1e-3, 1e3)`, compute pre/post NLL, Brier and 10-bin ECE on validation only, and reject if post-calibration NLL is non-finite or worse by more than `1e-9`. Store under checkpoint metadata:

```python
metadata["closed_set_calibration"] = calibration.to_dict()
metadata["experimental_closed_set"] = True
metadata["open_set_calibrated"] = False
```

Do not mark existing `metadata["calibration"]` as calibrated and do not fabricate thresholds.

- [ ] **Step 4: Add mutually exclusive calibration CLI paths**

For `--mode closed-set-temperature`, require `--known-val`, forbid `--unknown`, collect only authenticity logits/targets, verify the known validation SHA-256 equals the candidate metadata fingerprint, write a new candidate atomically, and print calibration metrics as UTF-8 JSON. Preserve the existing open-set CLI defaults and tests.

- [ ] **Step 5: Run calibration regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_calibration.py tests/unit/test_closed_set_calibration.py tests/integration/test_calibrate_cli.py tests/integration/test_closed_set_calibrate_cli.py -q`

Expected: all tests pass; closed-set output says `experimental_closed_set=true`, `open_set_calibrated=false`, and contains only one authenticity temperature.

- [ ] **Step 6: Commit closed-set calibration**

```powershell
git add 项目/Files/herb_authentication/inference/closed_set.py 项目/Files/herb_authentication/inference/calibration.py 项目/Files/herb_authentication/inference/__init__.py 项目/Files/calibrate.py tests/unit/test_closed_set_calibration.py tests/integration/test_closed_set_calibrate_cli.py tests/integration/test_calibrate_cli.py
git commit -m "feat: calibrate closed-set authenticity probabilities"
```

---

### Task 5: Complete Validation and Locked-Test Evaluation

**Files:**
- Modify: `项目/Files/herb_authentication/evaluation.py:24-430`
- Modify: `项目/Files/evaluate.py:19-245`
- Test: `tests/unit/test_authenticity_metrics.py`
- Modify test: `tests/unit/test_evaluation.py`
- Modify test: `tests/unit/test_evaluation_collection.py`
- Create: `tests/integration/test_partition_evaluation.py`
- Modify test: `tests/integration/test_evaluate_cli.py`
- Modify test: `tests/integration/test_evaluate_fingerprint.py`

**Interfaces:**
- Produces: `ClassMetrics(precision: float, recall: float, f1: float, support: int)`；`AuthenticityMetrics(confusion_matrix: tuple[tuple[int, int], tuple[int, int]], macro_f1: float, balanced_accuracy: float, per_class: Mapping[str, ClassMetrics], auroc: float, pr_auc: float, brier: float, ece: float, nll: float)`；`compute_authenticity_metrics(probabilities, targets, class_names=("authentic","counterfeit"), bins=10) -> AuthenticityMetrics`；`bootstrap_authenticity_metrics(..., samples: int, seed: int) -> Mapping[str, BootstrapInterval]`；`reserve_locked_test(run_dir, fingerprint_bundle) -> Path`。
- Consumes: Task 4 `ClosedSetCalibration`; existing `collect_known_predictions` and split assignment.

- [ ] **Step 1: Write failing exact-metric tests**

```python
def test_authenticity_metrics_report_confusion_per_class_and_probability_quality() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
    targets = torch.tensor([0, 0, 1, 1])
    result = compute_authenticity_metrics(probabilities, targets, bins=2)
    assert result.confusion_matrix == ((1, 1), (1, 1))
    assert result.macro_f1 == pytest.approx(0.5)
    assert result.balanced_accuracy == pytest.approx(0.5)
    assert result.per_class["counterfeit"].recall == pytest.approx(0.5)
    assert 0.0 <= result.auroc <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0
    assert result.brier >= 0.0
    assert result.ece >= 0.0
    assert result.nll >= 0.0
```

Add deterministic bootstrap tests for every required scalar, a two-class requirement, finite-value rejection, and `reserve_locked_test` refusing a second reservation even when arguments match.

- [ ] **Step 2: Run evaluation tests and verify red state**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_authenticity_metrics.py tests/unit/test_evaluation.py tests/integration/test_partition_evaluation.py -q`

Expected: missing metrics and partition/ledger interfaces fail.

- [ ] **Step 3: Implement full binary and calibration reporting**

Use scikit-learn only for `roc_auc_score` and `average_precision_score`; compute confusion, per-class precision/recall/F1, macro F1, balanced accuracy, Brier, ECE and NLL explicitly so class ordering is fixed. Report both uncalibrated and calibrated NLL/Brier/ECE; add 95% sample bootstrap intervals for Macro F1, Balanced Accuracy, per-class recall/F1, AUROC, PR-AUC, Brier, ECE and NLL.

- [ ] **Step 4: Make evaluation partition explicit and fail closed**

Add `--partition validation|locked-test`, `--run-dir`, and `--checkpoint`. Validation reads only `val.csv`; locked-test reads only `test.csv`, atomically creates `locked_test/reservation.json` before image inference, and refuses a second attempt. Both paths verify manifest, split, config, candidate and calibration fingerprints; any mismatch exits 2 without overwriting a prior report.

The report root must include:

```json
{
  "schema_version": 2,
  "evaluation_scope": "closed_set_authenticity",
  "partition": "validation",
  "experimental_closed_set": true,
  "open_set_calibrated": false,

  "limitations": [
    "仅适用于本数据集定义的骨碎补正品/伪品范围",
    "缺少批次、来源、设备和拍摄会话元数据，无法排除采集捷径",
    "伪品实际物种未检验",
    "没有 unknown calibration/test 数据"
  ]
}
```

Only the validation report computes `promotion_gate`; the locked-test report records whether the same thresholds are met but cannot feed trainer or calibration state.

- [ ] **Step 5: Run evaluation regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_authenticity_metrics.py tests/unit/test_evaluation.py tests/unit/test_evaluation_collection.py tests/integration/test_evaluate_cli.py tests/integration/test_evaluate_fingerprint.py tests/integration/test_partition_evaluation.py tests/integration/test_release_gates.py -q`

Expected: all tests pass; a second locked-test invocation exits 2 with `该 run 已执行 locked test` and leaves the first JSON/CSV unchanged.

- [ ] **Step 6: Commit evaluation and locked-test ledger**

```powershell
git add 项目/Files/herb_authentication/evaluation.py 项目/Files/evaluate.py tests/unit/test_authenticity_metrics.py tests/unit/test_evaluation.py tests/unit/test_evaluation_collection.py tests/integration/test_partition_evaluation.py tests/integration/test_evaluate_cli.py tests/integration/test_evaluate_fingerprint.py tests/integration/test_release_gates.py
git commit -m "feat: report locked closed-set evaluation metrics"
```

---

### Task 6: Experimental Path/Bytes Inference Boundary

**Files:**
- Modify: `项目/Files/herb_authentication/inference/closed_set.py`
- Modify: `项目/Files/herb_authentication/inference/service.py:33-304`
- Modify: `项目/Files/herb_authentication/inference/__init__.py`
- Modify: `项目/Files/predict.py:17-76`
- Create: `tests/unit/test_closed_set_inference.py`
- Modify test: `tests/integration/test_inference_service.py`
- Modify test: `tests/integration/test_predict_cli.py`
- Modify test: `tests/integration/test_cli_encoding.py`

**Interfaces:**
- Produces: `format_closed_set_result(result: ClosedSetPredictionResult) -> str`；`ClosedSetPredictionResult.to_dict() -> dict[str, Any]`; `InferenceService.predict_experimental_closed_set(image: Path | bytes) -> ClosedSetPredictionResult`；strict checkpoint metadata field `counterfeit_species_verified: bool` (false for this dataset)。
- Preserves: `InferenceService.predict_image(...) -> PredictionResult` and the existing formal open-set rejection behavior.
- Test helpers in `test_closed_set_inference.py`: `write_jpeg(path: Path) -> Path`; `make_closed_set_service(authenticity_logits: tuple[float, float], counterfeit_species_verified: bool) -> InferenceService`; `make_closed_set_result(authenticity: str, probability: float, actual_species_verified: bool) -> ClosedSetPredictionResult`.

- [ ] **Step 1: Write failing Path/bytes and wording tests**

```python
def test_closed_set_path_and_bytes_return_same_json(tmp_path: Path) -> None:
    image = write_jpeg(tmp_path / "sample.jpg")
    service = make_closed_set_service(authenticity_logits=(4.0, 0.0), counterfeit_species_verified=True)
    assert service.predict_experimental_closed_set(image).to_dict() == service.predict_experimental_closed_set(image.read_bytes()).to_dict()


def test_authentic_closed_set_message_has_only_verdict_and_probability() -> None:
    result = make_closed_set_result(authenticity="authentic", probability=0.91, actual_species_verified=True)
    assert result.message == "此药品为正品，有91.0%的可能该药材为正品。"
    assert "声称为" not in result.message
    assert "实际为" not in result.message


def test_unverified_actual_species_suppresses_counterfeit_verdict() -> None:
    result = make_closed_set_result(authenticity="counterfeit", probability=0.93, actual_species_verified=False)
    assert result.message == "实际药材无法确定，有93.0%的可能该药材为伪品，请人工复核。"
    assert result.authenticity.label == "counterfeit"
    assert result.authenticity.confidence == pytest.approx(0.93)
```

Also cover the verified counterfeit template `此药品为伪品，有93.0%的可能该药材为伪品，声称为‘当归’，实际应为‘欧当归’。`, JSON serialization, exact flags, warning, hashes, corrupt/oversized bytes, and missing closed-set calibration.

- [ ] **Step 2: Run inference tests and verify red state**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_closed_set_inference.py tests/integration/test_inference_service.py tests/integration/test_predict_cli.py -q`

Expected: missing `predict_experimental_closed_set` and response type failures.

- [ ] **Step 3: Implement a versioned closed-set response**

```python
@dataclass(frozen=True, slots=True)
class ClosedSetPredictionResult:
    model_version: str
    authenticity: ScoredLabel
    probabilities: Mapping[str, float]
    actual_species_verified: bool
    message: str
    model_fingerprint: str
    data_fingerprint: str
    split_fingerprint: str
    experimental_closed_set: bool = True
    open_set_calibrated: bool = False
    scope_warning: str = "仅适用于本数据集定义的骨碎补正品/伪品范围"
    schema_version: int = 1
```

`to_dict` must copy mappings to plain `dict`, cast all numeric values to `float`, keep Chinese Unicode, and validate probabilities are finite, sum to 1 within `1e-6`, and match `authenticity.confidence`.

- [ ] **Step 4: Reuse the safe decode path and apply only the frozen authenticity temperature**

`predict_experimental_closed_set` must call `predict_logits`, divide only authenticity logits by `ClosedSetCalibration.temperature`, and return both probabilities. It must not call or imply `apply_rejection_policy` because no unknown thresholds exist. The formal `predict_image` method remains unchanged and continues to reject uncalibrated/open-set-unsafe results.

- [ ] **Step 5: Add explicit CLI opt-in**

`predict.py --experimental-closed-set --json --image 项目/References/raw/Drynaria_fortunei_authentic/Drynaria-fortunei_AUTHENTIC_0001.jpg` calls the new method. Require `--json` for experimental mode so future Web code consumes structured fields; missing calibration or fingerprints exits 2. Default CLI behavior remains backward compatible.

- [ ] **Step 6: Run inference and encoding regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_closed_set_inference.py tests/unit/test_result_formatter.py tests/integration/test_inference_service.py tests/integration/test_inference_logits.py tests/integration/test_predict_cli.py tests/integration/test_cli_encoding.py -q`

Expected: all tests pass; stdout decodes as UTF-8 without `PYTHONUTF8`; experimental JSON always contains both status flags and the scope warning.

- [ ] **Step 7: Commit the experimental inference boundary**

```powershell
git add 项目/Files/herb_authentication/inference/closed_set.py 项目/Files/herb_authentication/inference/service.py 项目/Files/herb_authentication/inference/__init__.py 项目/Files/predict.py tests/unit/test_closed_set_inference.py tests/integration/test_inference_service.py tests/integration/test_predict_cli.py tests/integration/test_cli_encoding.py
git commit -m "feat: expose experimental closed-set inference"
```

---

### Task 7: Safe Candidate Promotion

**Files:**
- Create: `项目/Files/herb_authentication/training/promotion.py`
- Create: `项目/Files/promote.py`
- Modify: `项目/Files/herb_authentication/training/__init__.py`
- Create: `tests/unit/test_promotion.py`
- Modify test: `tests/integration/test_release_gates.py`

**Interfaces:**
- Produces: `PromotionError(HerbAuthenticationError)`；`PromotionThresholds(macro_f1: float, balanced_accuracy: float, counterfeit_recall: float)`；`DEFAULT_PROMOTION_THRESHOLDS = PromotionThresholds(0.80, 0.80, 0.80)`；`PromotionRequest(candidate: Path, target: Path, validation_report: Path, locked_test_report: Path, run_dir: Path)`；`PromotionRecord(old_sha256: str | None, new_sha256: str, target: Path, run_id: str, promoted_at: str)`；`promote_candidate(request, thresholds, smoke_image: Path) -> PromotionRecord`。
- Consumes: Task 5 report schema v2; Task 6 checkpoint loader and experimental Path/bytes inference.
- Test helpers in `test_promotion.py`: `make_request(root: Path, *, macro_f1: float, balanced_accuracy: float, counterfeit_recall: float) -> PromotionRequest`; `passing_request(root: Path) -> PromotionRequest`; both create schema-v1 candidate checkpoints, schema-v2 reports, a smoke JPEG and an optional existing target with matching fingerprints.

- [ ] **Step 1: Write failing gate and atomicity tests**

```python
def test_promotion_rejects_below_threshold_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "official.pth"
    target.write_bytes(b"old-model")
    request = make_request(tmp_path, macro_f1=0.79, balanced_accuracy=0.90, counterfeit_recall=0.90)
    with pytest.raises(PromotionError, match="macro_f1"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, smoke_image=tmp_path / "image.jpg")
    assert target.read_bytes() == b"old-model"


def test_successful_promotion_records_old_and_new_hashes(tmp_path: Path) -> None:
    request = passing_request(tmp_path)
    old_hash = sha256_file(request.target)
    candidate_hash = sha256_file(request.candidate)
    smoke_image = tmp_path / "smoke.jpg"
    record = promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, smoke_image=smoke_image)
    assert record.old_sha256 == old_hash
    assert record.new_sha256 == candidate_hash
    assert sha256_file(request.target) == record.new_sha256
```

Also reject NaN/Inf, fingerprint disagreement, missing locked-test reservation/report, `open_set_calibrated=true`, absent experimental flag, unsupported schema, load/smoke failure, target/candidate same path, and simulated `os.replace` failure.

- [ ] **Step 2: Run tests and verify red state**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_promotion.py tests/integration/test_release_gates.py -q`

Expected: missing promotion module/CLI failures.

- [ ] **Step 3: Implement validation and test gates before any write**

Require both reports to share run ID, candidate hash, manifest hash, split hash, config hash and calibration hash. Require validation to pass all three preregistered thresholds; require locked test to be complete and meet the same three thresholds. Safely load the candidate via `InferenceService.from_checkpoint`, run both Path and bytes experimental inference on the same smoke image, and compare their JSON values.

- [ ] **Step 4: Implement atomic promotion and audit record**

Copy candidate bytes to a same-directory temporary file, verify temporary SHA-256 and safe load, `os.replace` it onto the target, then atomically write `promotion.json` containing run ID, UTC time, candidate/target paths, previous hash or null, new hash, thresholds, validation/test metric values, all fingerprints, and `schema_version=1`. If any pre-replace check fails, target is untouched; if report writing fails after replacement, emit an explicit recovery error containing both hashes.

- [ ] **Step 5: Add promotion CLI**

`promote.py --config ... --run-dir ... --candidate ... --smoke-image ...` gets the target and thresholds from typed config, configures UTF-8, prints the promotion record as JSON, and exits 2 on any gate failure. It never trains, calibrates, or evaluates.

- [ ] **Step 6: Run promotion regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_promotion.py tests/integration/test_release_gates.py tests/unit/test_checkpoints.py -q`

Expected: all tests pass; old targets survive all simulated failures byte-for-byte.

- [ ] **Step 7: Commit safe promotion**

```powershell
git add 项目/Files/herb_authentication/training/promotion.py 项目/Files/herb_authentication/training/__init__.py 项目/Files/promote.py tests/unit/test_promotion.py tests/integration/test_release_gates.py
git commit -m "feat: gate and atomically promote candidate models"
```

---

### Task 8: End-to-End Synthetic Regression and Operator Documentation

**Files:**
- Create: `tests/integration/test_closed_set_pipeline.py`
- Modify: `使用指南.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Tasks 1–7 CLIs and public APIs.
- Produces: `TinyPipelineArtifacts(training_ids: frozenset[str], validation_ids: frozenset[str], test_ids: frozenset[str], calibration_ids: frozenset[str], evaluation_ids: frozenset[str], response: Mapping[str, object])` and `run_tiny_pipeline(root: Path) -> TinyPipelineArtifacts`; a fast CPU synthetic pipeline proving boundaries and an operator runbook with exact commands.

- [ ] **Step 1: Write a failing synthetic end-to-end test**

Create six tiny RGB images per class, inject a tiny encoder/model factory, and run the public functions in order. Assertions must cover: 12 unique sample/group IDs; each class in every partition; fake species masks false; train loader never contains val/test IDs; calibration sees val IDs only; locked-test reservation exists before test inference; report flags are exact; promotion changes only the requested target.

```python
def test_closed_set_pipeline_keeps_train_val_test_roles_isolated(tmp_path: Path) -> None:
    artifacts = run_tiny_pipeline(tmp_path)
    assert artifacts.training_ids.isdisjoint(artifacts.validation_ids)
    assert artifacts.training_ids.isdisjoint(artifacts.test_ids)
    assert artifacts.calibration_ids == artifacts.validation_ids
    assert artifacts.evaluation_ids == artifacts.test_ids
    assert artifacts.response["experimental_closed_set"] is True
    assert artifacts.response["open_set_calibrated"] is False
```

- [ ] **Step 2: Run the new test and verify the first missing integration fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_closed_set_pipeline.py -q`

Expected: FAIL until all public interfaces from Tasks 1–7 are wired consistently.

- [ ] **Step 3: Complete wiring without adding a second orchestration framework**

Fix only integration gaps discovered by the test. Reuse existing CLIs/public functions; do not add a workflow engine, HTTP server, database, frontend, or new model family.

- [ ] **Step 4: Update operator documentation**

Document environment preflight, data requirements, exact commands, artifact locations, output wording, promotion gates, one-time locked-test rule, encoding behavior, known limitations, and future Web pseudocode:

```python
service = InferenceService.from_checkpoint(MODEL_PATH, device="cpu")
payload = uploaded_file.read()
response = service.predict_experimental_closed_set(payload).to_dict()
return JSONResponse(response)
```

State explicitly that upload size/MIME/authentication/rate limiting belong to the future HTTP adapter and do not weaken `InferenceService` byte/pixel/decode limits.

- [ ] **Step 5: Run the complete automated quality gate**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: the prior 210 tests plus all new tests pass; no skips or xfails introduced for this feature.

Run: `.\.venv\Scripts\python.exe -m ruff check 项目\Files tests`

Expected: `All checks passed!`

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 6: Commit integration and docs**

```powershell
git add tests/integration/test_closed_set_pipeline.py 使用指南.md .gitignore
git commit -m "docs: add closed-set operation runbook"
```

---

### Task 9A: Versioned Near-Duplicate-Component Constrained Splitter

**Files:**
- Modify: `docs/superpowers/specs/2026-08-09-closed-set-authentication-completion-design.md`
- Modify: `docs/superpowers/plans/2026-08-09-closed-set-authentication-completion.md`
- Modify: `项目/Files/herb_authentication/config.py`
- Modify: `项目/Files/config_closed_set.yaml`
- Modify: `项目/Files/herb_authentication/data/splitting.py`
- Modify: `项目/Files/herb_authentication/data/__init__.py`
- Modify: `项目/Files/prepare_closed_set.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_splitting.py`
- Test: `tests/integration/test_prepare_closed_set_cli.py`
- Test: `tests/integration/test_train_assembly.py`
- Test: `tests/integration/test_evaluate_fingerprint.py`

**Interfaces:**
- Consumes: `ValidatedKnownRecord.perceptual_hash`; v1 `create_group_splits(...)`/`apply_split(...)`; fixed real targets train `(235,64)`, val `(50,14)`, test `(50,14)`.
- Produces: `create_group_splits(..., algorithm_version: str = "group-stratified-v1", target_counts: Mapping[str, Mapping[str, int]] | None = None) -> SplitAssignment`; v2 auditable `SplitAssignment`; atomic prepare suite.

- [ ] **Step 1: Update approved spec and plan before code**

Record v1 compatibility, v2 all-pairs inclusive graph/transitive closure, exact DP, secondary objective, stable tie-break, audit schema, independent oracle, atomic suite publication, check-only behavior, and Task 9A hard stop.

- [ ] **Step 2: RED — graph semantics, IDs, determinism, exact solver**

Add focused tests to `tests/unit/test_splitting.py`:

- `test_v2_threshold_is_inclusive_at_four_but_not_five`: construct hashes at Hamming 4 and 5; assert distance-4 groups share a component and distance-5 groups do not.
- `test_v2_transitive_near_duplicate_chain_stays_atomic`: A-B and B-C edges with A-C >4; assert A/B/C share partition.
- `test_v2_mapping_is_independent_of_input_order`: reverse rows; assert `by_group`, component mapping/fingerprint, objective and mapping fingerprint identical.
- `test_v2_preserves_unique_sample_and_group_ids`: apply assignment; assert original 1:1 IDs unchanged and exactly once.
- `test_v2_exactly_solves_same_and_mixed_label_components`: supply explicit targets; assert all six partition×label cells exact and mixed components atomic.
- `test_v2_rejects_unsatisfiable_exact_targets`: build component vector larger than every compatible target; assert `ManifestValidationError` mentions no exact solution.
- `test_v2_does_not_concentrate_all_multi_components_in_train`: construct multiple feasible optima; assert secondary objective places multi-member label mass according to target proportions.
- `test_v2_independent_all_pairs_oracle_reports_zero_crossings`: independently enumerate every pair; assert no cross-partition pair has Hamming `<=4`.

Run: `.\.venv\python.exe -m pytest tests\unit\test_splitting.py -q`

Expected RED: `create_group_splits()` rejects missing `algorithm_version`/`target_counts`, or v2 constant/API is absent.

- [ ] **Step 3: GREEN — minimal graph/component/exact-DP implementation**

Implement v2 without changing v1 mapping. Graph nodes remain original group IDs. Build all image pairs, union group nodes at Hamming `<= threshold`, canonicalize components by sorted group IDs, and calculate per-component two-label vectors. Exact DP/backtracking enforces every target cell; no solution raises `ManifestValidationError`. Among feasible solutions minimize the spec objective, then apply stable seed/component hash and mapping-order tie-break. Singleton fill may optimize state size but must remain auditable and exact.

Repair invariant: quota state must retain objective-relevant multi distribution; no early hash pruning across different distributions. Exact solver may decompose mixed-label coupled DP + per-label weighted DPs only under §5.3 additivity/disjoint-lex proof. Verify reviewer counterexample objective=16, input reverse, same-vector paths, hash collision, independent exhaustive oracle. Formal-vector read-only target: <30s; record state counts/time.

Run the Step 2 command. Expected: new v2 tests pass and all existing v1 splitting tests remain green.

- [ ] **Step 4: RED/GREEN — complete audit and tamper-proof replay**

Add tests that serialize v2 then independently mutate each critical family: algorithm/hash threshold semantics, component version/count/mapping/fingerprint, solver version/status/objective/tie-break, target/actual counts, input/mapping fingerprints, `by_group`, oracle version/pair/cross count. Each mutation must make `apply_split()` fail closed. Verify v1 and v2 both replay; v1 JSON remains schema-compatible.

Run: `.\.venv\python.exe -m pytest tests\unit\test_splitting.py -q`

Expected: all audit/tamper cases pass.

- [ ] **Step 5: RED/GREEN — config routing and train/evaluate compatibility**

Add `DataConfig.split_algorithm_version` defaulting to `group-stratified-v1`; allow exactly v1/v2. Formal `config_closed_set.yaml` explicitly selects `group-stratified-near-duplicate-components-v2`. Update train/evaluate split readers only as needed so `apply_split()` accepts both persisted versions; no training/evaluation command may run in Task 9A.

Repair invariant: V2 missing split or config/assignment version mismatch → fail before run/checkpoint/report/logits. Missing-split fallback ∈ V1 only. Train/evaluate config fingerprint includes split version and remains byte-identical for same config.

Run: `.\.venv\python.exe -m pytest tests\unit\test_config.py tests\integration\test_train_assembly.py tests\integration\test_evaluate_fingerprint.py -q`

Expected: v1 default and v2 explicit routing pass; unsupported versions fail closed.

- [ ] **Step 6: RED/GREEN — atomic prepare and check-only**

Update `tests/integration/test_prepare_closed_set_cli.py` for v2 target matrix, full audit report, failure preserving an existing six-file suite, and `--check-only` zero writes. Prepare must build all outputs in one staging directory, validate replay/oracle/hash, then publish with rollback on any replacement failure.

Run: `.\.venv\python.exe -m pytest tests\integration\test_prepare_closed_set_cli.py -q`

Expected: v2 prepare passes; count/no-solution/publish failure leaves the complete prior suite byte-identical.

- [ ] **Step 7: Quality gate and exact commit**

Run:

```powershell
.\.venv\python.exe -m pytest tests\unit\test_splitting.py tests\unit\test_config.py tests\integration\test_prepare_closed_set_cli.py tests\integration\test_train_assembly.py tests\integration\test_evaluate_fingerprint.py -q
.\.venv\python.exe -m pytest -q
.\.venv\python.exe -m ruff check 项目\Files tests
.\.venv\python.exe -m ruff format --check 项目\Files\herb_authentication\config.py 项目\Files\herb_authentication\data\splitting.py 项目\Files\herb_authentication\data\__init__.py 项目\Files\prepare_closed_set.py tests\unit\test_config.py tests\unit\test_splitting.py tests\integration\test_prepare_closed_set_cli.py tests\integration\test_train_assembly.py tests\integration\test_evaluate_fingerprint.py
git diff --check
```

Stage only code/tests/spec/plan/config; exclude generated real metadata/splits/partition CSV and all unrelated dirty files. Commit without amend.

- [ ] **Step 8: Run formal prepare once, audit, and stop**

Run: `.\.venv\python.exe 项目\Files\prepare_closed_set.py --config 项目\Files\config_closed_set.yaml`

Expected: 427 unique sample IDs, 427 unique group IDs; 335/92; train 235/64, val 50/14, test 50/14; algorithm v2; all component/solver/fingerprint fields complete; independent full-pair final cross count 0. Update SDD `task-9-report.md` and ledger. Do not stage generated real artifacts. Stop before train/evaluate/locked-test/promotion.

---

### Task 9: Execute the Real 427-Image Experiment and Promote Only if Eligible

**Files:**
- Generate and track: `项目/Data/experiments/drynaria_closed_set/metadata.csv`
- Generate and track: `项目/Data/experiments/drynaria_closed_set/splits.json`
- Generate but ignore: `项目/Data/experiments/drynaria_closed_set/train.csv`
- Generate but ignore: `项目/Data/experiments/drynaria_closed_set/val.csv`
- Generate but ignore: `项目/Data/experiments/drynaria_closed_set/test.csv`
- Generate but ignore: `项目/Models/experiments/drynaria_closed_set.candidate.pth`
- Generate but ignore: `项目/Models/experiments/drynaria_closed_set.calibrated.candidate.pth`
- Generate but ignore: `项目/Results/experiments/drynaria_closed_set/runs/20260825-drynaria-closed-set-v2/...`
- Retain but do not target: `项目/Results/experiments/drynaria_closed_set/runs/20260809-drynaria-closed-set-v1/...`（在 epoch 0 前失败的旧 run）
- Create and track: `docs/experiments/2026-08-09-drynaria-closed-set.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: immutable real run artifacts, a calibrated candidate, one locked-test report, optional promoted official checkpoint, and a human-readable experiment summary.

- [ ] **Step 1: Record environment and repository preflight**

Run:

```powershell
$runId = "20260825-drynaria-closed-set-v2"
$runDir = Join-Path 项目\Results\experiments\drynaria_closed_set\runs $runId
.\.venv\python.exe 项目\Files\doctor.py
.\.venv\python.exe -c "import sys,torch,torchvision; print(sys.version); print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
git status --short
```

Expected: Python is 3.11 or 3.12; torch/torchvision import; existing unrelated dirty files are recorded and not staged. Keep this PowerShell session and the same `$runId`/`$runDir` values for Steps 3–7.

- [ ] **Step 2: Generate and validate the real manifest and locked split once**

Run: `.\.venv\python.exe 项目\Files\prepare_closed_set.py --config 项目\Files\config_closed_set.yaml`

Expected: exit 0; manifest has 427 unique sample IDs and 427 unique group IDs; counts are exactly 335/92 and partitions exactly 299/64/64 with the specified class counts; validation report has no errors.

- [ ] **Step 3: Create a run ID and train only on train while selecting on validation**

Run: `.\.venv\python.exe 项目\Files\train.py --config 项目\Files\config_closed_set.yaml --device cpu --run-id $runId`

Expected: exit 0; stdout gives run ID and candidate path; checkpoint metadata hashes equal generated manifest/split/config; `training_history.json` has finite loss and metrics; no formal model path changes.

If validation stalls without a clear train/validation gap or other sign of overfitting, do not inspect test. Create one new versioned config that changes only the preregistered training choice (for example `unfreeze_epoch: 5` with encoder LR 0.1× head LR), start a new run and document the comparison using validation only. Never edit a completed run directory or reuse its checkpoint for incompatible data/config.

- [ ] **Step 4: Fit closed-set temperature using validation only**

Run:

```powershell
.\.venv\python.exe 项目\Files\calibrate.py --mode closed-set-temperature --checkpoint 项目\Models\experiments\drynaria_closed_set.candidate.pth --known-val 项目\Data\experiments\drynaria_closed_set\val.csv --output 项目\Models\experiments\drynaria_closed_set.calibrated.candidate.pth --data-root 项目\References --device cpu
```

Expected: exit 0; positive finite temperature; post NLL no worse than pre NLL; candidate metadata records validation/candidate hashes and both experimental/open-set flags.

- [ ] **Step 5: Evaluate validation and enforce the preregistered promotion gate**

Run: `.\.venv\python.exe 项目\Files\evaluate.py --config 项目\Files\config_closed_set.yaml --checkpoint 项目\Models\experiments\drynaria_closed_set.calibrated.candidate.pth --partition validation --run-dir $runDir --json`

Expected: report contains all required metrics and intervals. Continue to Step 6 only if Macro F1, Balanced Accuracy, and counterfeit recall are each at least 0.80 and fingerprints are complete. If not, retain the candidate/report, document the failed validation gate, and improve only through a new validation-only run; do not execute locked test and do not promote.

- [ ] **Step 6: Run locked test exactly once**

Run: `.\.venv\python.exe 项目\Files\evaluate.py --config 项目\Files\config_closed_set.yaml --checkpoint 项目\Models\experiments\drynaria_closed_set.calibrated.candidate.pth --partition locked-test --run-dir $runDir --json`

Expected: reservation and immutable test report are written once; report contains confusion matrix, Macro F1, Balanced Accuracy, per-class precision/recall/F1, AUROC, PR-AUC, Brier, ECE, pre/post NLL and bootstrap 95% intervals. Do not rerun even if metrics are disappointing.

- [ ] **Step 7: Run inference smoke checks and conditionally promote**

Read the first locked-test image path from the generated CSV, run experimental Path and bytes inference against those exact bytes, and confirm JSON equivalence. If validation and locked-test gates both pass, run:

```powershell
$smokePath = (Import-Csv -Encoding UTF8 项目\Data\experiments\drynaria_closed_set\test.csv | Select-Object -First 1).image_path
$smokeImage = if ([IO.Path]::IsPathRooted($smokePath)) {
    (Resolve-Path -LiteralPath $smokePath).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path 项目\References $smokePath)).Path
}
.\.venv\python.exe 项目\Files\promote.py --config 项目\Files\config_closed_set.yaml --run-dir $runDir --candidate 项目\Models\experiments\drynaria_closed_set.calibrated.candidate.pth --smoke-image $smokeImage
```

Expected on pass: target safely loads, old/new SHA-256 values are recorded, and `项目/Models/best_multitask_model.calibrated.pth` equals the candidate hash. Expected on gate failure: exit 2, formal target unchanged, candidate and reports retained.

- [ ] **Step 8: Write the real experiment summary without overstating capability**

Record exact run ID, commands, environment versions, 427/335/92 counts, split counts, all hashes, best epoch, selection score, temperature, validation metrics, locked-test metrics, confidence intervals, promotion result, limitations, and any warnings. Use “闭集实验” throughout; explicitly state no unknown rejection evidence and no counterfeit species identification.

- [ ] **Step 9: Re-run final regression and inspect tracked scope**

Run:

```powershell
.\.venv\python.exe -m pytest -q
.\.venv\python.exe -m ruff check 项目\Files tests
git diff --check
git status --short
```

Expected: tests and Ruff pass, diff check is clean, and only the new manifest, split, experiment summary and intentional source/test/doc changes are staged; raw images, model binaries, Results, prior positive-only edits and unrelated sibling-project changes remain unstaged.

- [ ] **Step 10: Commit only portable real-experiment records**

```powershell
git add 项目/Data/experiments/drynaria_closed_set/metadata.csv 项目/Data/experiments/drynaria_closed_set/splits.json docs/experiments/2026-08-09-drynaria-closed-set.md
git diff --cached --check
git commit -m "exp: record drynaria closed-set evaluation"
```

---

## Final Acceptance Checklist

- [ ] `metadata.csv` has exactly 427 rows, 427 unique `sample_id`, 427 unique `group_id`, 335 authentic and 92 counterfeit rows.
- [ ] Every authentic row has verified `Drynaria_fortunei`; every counterfeit row has blank/unverified actual species.
- [ ] `splits.json` deterministically reproduces 299/64/64 with seed 42 and contains the current input fingerprint.
- [ ] No exact or configured perceptual near duplicate crosses partitions; any detected issue blocks training.
- [ ] Best checkpoint selection uses only authenticity Macro F1 and Balanced Accuracy.
- [ ] Exactly one class-imbalance mechanism is active.
- [ ] Closed-set calibration uses validation authenticity logits only and preserves `open_set_calibrated=false`.
- [ ] Validation and locked-test reports contain every required discrimination/calibration metric and bootstrap interval.
- [ ] Each run can reserve locked test only once; test never affects training, selection, or temperature fitting.
- [ ] Experimental Path and bytes responses match, serialize to JSON, and contain the exact flags and warning.
- [ ] Authentic wording contains only verdict and probability; an unverified actual species uses the exact counterfeit-probability review wording.
- [ ] Candidate promotion is fail-closed, schema-compatible, hash-audited and atomic.
- [ ] All automated tests, Ruff, CLI encoding checks and `git diff --check` pass.
- [ ] The experiment summary states the data-source limitations and makes no open-set or counterfeit-species claim.
