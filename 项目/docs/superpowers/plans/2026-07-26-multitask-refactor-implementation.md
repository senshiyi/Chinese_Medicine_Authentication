# 中药识别三任务重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有单头图片分类项目重构为可维护、可复现、可校准的三任务系统，分别输出声称药材、实际物种和真伪状态，并通过稳定的 Python 服务接口为未来 Web 上传页面预留能力。

**Architecture:** 保留 `项目/Files` 下的 train/evaluate/predict 薄 CLI，新增独立 calibrate 与 doctor 入口，核心代码迁入 `项目/Files/herb_authentication` 分层包。数据层使用 manifest 与 `group_id` 固化切分，模型层共享骨干并提供三个任务头，训练层负责掩码损失、指标和可恢复 checkpoint，推理层集中完成校准、拒识、结构化返回及精确中文文案。环境预检在导入重量级依赖前运行，旧模型文件永不覆盖。

**Tech Stack:** Python 3.12、PyTorch、torchvision、Pillow、PyYAML、NumPy、scikit-learn、pytest、ruff。

## Global Constraints

- 所有源码、配置、JSON 和 Markdown 使用 UTF-8；面向 Excel 的 CSV 使用 UTF-8-SIG。外部文本有显式编码时严格使用该编码；未指定时先尝试 UTF-8-SIG，失败后才回退 GBK、发出警告并记录实际编码，禁止依赖系统默认编码。
- 当前 `项目/Models/best_model.pth` 与 `epoch_001.pth` 只读保留；新版 checkpoint 默认写入 `项目/Models/best_multitask_model.pth`。
- 测试不下载预训练权重、不访问网络；模型测试注入微型 encoder，DataLoader 使用 `num_workers=0`。
- 没有真实数据时只证明结构、流程、确定性和错误处理正确，不声称真实准确率。
- 只有完整校准且三个任务都达到阈值时才能输出正品或伪品；实际药材无法确定时必须整体拒识。
- 每完成一个任务，只提交该任务列出的文件；共享工作区中的无关改动不得暂存、修改或提交。
- 测试与 CLI 命令从项目目录 `E:\Desktop\WorkSpace\Chinese_Medicine_Authentication` 运行；每个 Commit 代码块从父 Git 根目录 `E:\Desktop\WorkSpace` 运行，避免二重项目路径。

---

### Task 1: 建立可安装包、隔离环境说明与标准库预检

**Files:**
- Create: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `项目/Files/herb_authentication/__init__.py`
- Create: `项目/Files/herb_authentication/errors.py`
- Create: `项目/Files/herb_authentication/environment.py`
- Create: `项目/Files/doctor.py`
- Test: `tests/unit/test_environment.py`
- Test: `tests/integration/test_doctor_cli.py`

**Interfaces:**
- Produce: `EnvironmentReport`, `DependencyProbe`, `run_bootstrap_preflight()`, `run_dependency_probe()`。
- Produce: `ConfigurationError`、`ManifestValidationError`、`EnvironmentCompatibilityError`、`InputImageError`、`CheckpointCompatibilityError`，全部从 `herb_authentication.errors` 统一导入。
- Constraint: `environment.py` 顶层只能导入 Python 标准库；重量级依赖在隔离子进程内探测。

- [x] **Step 0: 建立最小可安装骨架与隔离环境**

先创建 `pyproject.toml`、锁定的 `requirements.txt`、空包 `herb_authentication/__init__.py` 和包含全部领域异常的 `errors.py`；此时不要创建 `environment.py`。`pyproject.toml` 设置 `requires-python = ">=3.11,<3.13"`，把非标准包根目录和 pytest `pythonpath` 都配置为 `项目/Files`。

Run: `conda create --prefix '.venv' python=3.12 pip -y`

Run: `.venv\python.exe -m pip install -r requirements.txt -e .`

Expected: 隔离环境创建成功，pytest/ruff 与项目空包可导入；不使用当前混装的 Anaconda/PyTorch 环境。

- [x] **Step 1: 写失败测试**

```python
from herb_authentication.environment import run_bootstrap_preflight


def test_bootstrap_preflight_is_stdlib_only_and_reports_utf8():
    report = run_bootstrap_preflight()
    assert report.python_supported is True
    assert report.internal_text_encoding == "utf-8"
    assert report.imported_heavy_modules == ()
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_environment.py -q`

Expected: FAIL，`herb_authentication.environment` 尚不存在。

- [x] **Step 3: 实现最小预检与依赖探测**

```python
@dataclass(frozen=True)
class EnvironmentReport:
    python_version: str
    python_supported: bool
    preferred_encoding: str
    internal_text_encoding: str
    filesystem_encoding: str
    imported_heavy_modules: tuple[str, ...]


def run_bootstrap_preflight() -> EnvironmentReport:
    heavy = ("torch", "torchvision", "numpy", "sklearn")
    return EnvironmentReport(
        python_version=platform.python_version(),
        python_supported=(3, 11) <= sys.version_info[:2] < (3, 13),
        preferred_encoding=locale.getpreferredencoding(False),
        internal_text_encoding="utf-8",
        filesystem_encoding=sys.getfilesystemencoding(),
        imported_heavy_modules=tuple(name for name in heavy if name in sys.modules),
    )
```

`requirements.txt` 锁定一组相互兼容的 CPU 基线依赖；GPU 依赖作为文档化替换方案，不混装。`run_dependency_probe()` 使用 `subprocess.run([sys.executable, "-I", "-c", probe_code], encoding="utf-8", errors="replace")` 探测 torch/torchvision/numpy/sklearn/Pillow 版本、CUDA 状态与导入错误。`doctor.py` 只在函数体内导入项目包并以 UTF-8 JSON 输出，异常退出码为 `2`。

- [x] **Step 4: 验证 GREEN 和 CLI**

Run: `.venv\python.exe -m pytest tests/unit/test_environment.py -q`

Expected: PASS。

Run: `.venv\python.exe 项目/Files/doctor.py --json`

Expected: 输出合法 UTF-8 JSON；依赖缺失或冲突时返回非零并给出隔离环境建议，而不是直接崩溃。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/.gitignore Chinese_Medicine_Authentication/pyproject.toml Chinese_Medicine_Authentication/requirements.txt Chinese_Medicine_Authentication/项目/Files/herb_authentication Chinese_Medicine_Authentication/项目/Files/doctor.py Chinese_Medicine_Authentication/tests/unit/test_environment.py Chinese_Medicine_Authentication/tests/integration/test_doctor_cli.py
git commit -m "build: 建立隔离环境与启动预检"
```

---

### Task 2: 实现显式编码策略与类型化配置

**Files:**
- Create: `项目/Files/herb_authentication/encoding.py`
- Create: `项目/Files/herb_authentication/config.py`
- Modify: `项目/Files/config.yaml`
- Test: `tests/unit/test_encoding.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consume: YAML 配置路径和可选 `--text-encoding` 覆盖。
- Produce: `AppConfig` 及嵌套的 `DataConfig`、`ModelConfig`、`TrainingConfig`、`CalibrationConfig`、`InferenceConfig`。
- Produce: `DecodedText(text, actual_encoding, used_fallback)`、`read_text(path, encoding=None)`、`write_json_utf8(path, value)`、`write_csv_utf8_sig(path, rows)`。

- [x] **Step 1: 写编码与未知键失败测试**

```python
def test_read_text_supports_explicit_gbk(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_bytes("当归".encode("gbk"))
    assert read_text(path, encoding="gbk").text == "当归"


def test_default_read_falls_back_to_gbk_and_records_warning(tmp_path, caplog):
    path = tmp_path / "labels.txt"
    path.write_bytes("当归".encode("gbk"))
    decoded = read_text(path)
    assert decoded.text == "当归"
    assert decoded.actual_encoding == "gbk"
    assert decoded.used_fallback is True
    assert "GBK" in caplog.text


def test_explicit_encoding_never_silently_falls_back(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_text(path, encoding="utf-8")


def test_json_keeps_chinese_and_csv_has_bom(tmp_path):
    write_json_utf8(tmp_path / "result.json", {"label": "当归"})
    write_csv_utf8_sig(tmp_path / "result.csv", [{"label": "当归"}])
    assert "当归" in (tmp_path / "result.json").read_text(encoding="utf-8")
    assert (tmp_path / "result.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_config_rejects_unknown_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  unknow_key: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknow_key"):
        load_config(path)
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_encoding.py tests/unit/test_config.py -q`

Expected: FAIL，接口尚不存在。

- [x] **Step 3: 实现严格配置加载**

配置 dataclass 必须提供默认值并验证：比例和概率在 `[0, 1]`、三切分比例和为 `1`、batch/epoch/image size 为正数、编码只允许 `utf-8`、`utf-8-sig`、`gbk`，`data.provenance` 只允许 `unknown/synthetic/real` 且默认 `unknown`。YAML 使用 `yaml.safe_load`，递归检查未知键，路径相对配置文件解析并标准化为绝对 `Path`。未指定文本编码时先按 UTF-8-SIG，解码失败后才以 GBK 回退并警告；显式编码失败不得回退。实际编码与 provenance 写入 validation report、checkpoint 和 evaluation report；JSON 固定 `ensure_ascii=False`，CSV 固定 UTF-8-SIG。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_encoding.py tests/unit/test_config.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/encoding.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/config.py Chinese_Medicine_Authentication/项目/Files/config.yaml Chinese_Medicine_Authentication/tests/unit/test_encoding.py Chinese_Medicine_Authentication/tests/unit/test_config.py
git commit -m "feat: 增加严格配置与中文编码策略"
```

---

### Task 3: 定义 manifest、标签语义与数据校验

**Files:**
- Create: `项目/Files/herb_authentication/data/__init__.py`
- Create: `项目/Files/herb_authentication/data/manifest.py`
- Test: `tests/unit/test_manifest.py`
- Test: `tests/fixtures/manifest_minimal.csv`
- Test: `tests/fixtures/unknown_manifest_minimal.csv`
- Test: `tests/fixtures/validation_report_expected.json`

**Interfaces:**
- `KnownManifestRow(image_path, group_id, sample_id, claimed_herb_name, authenticity, actual_species, species_annotation_status, batch_id, origin, capture_device, collected_at, annotator, notes)` 完全对应 `metadata.csv`。
- `UnknownManifestRow(image_path, group_id, sample_id, unknown_type, dataset_role)` 完全对应独立的 `unknown_metadata.csv`；`dataset_role` 只允许 `calibration/test`。
- `actual_species` 在 CSV 中使用 `|` 分隔并解析为 `tuple[str, ...]`；只有 `species_annotation_status=verified` 的行参与物种监督。
- `ValidatedImageRecord` 增加图片 SHA-256、可选感知哈希、规范化标签及已解码编码，供切分和 checkpoint 指纹使用。
- `load_known_manifest()`、`load_unknown_manifest()` 分别解析；`validate_manifest_pair()` 生成排序后的 UTF-8 `validation_report.json` 并在存在错误时抛出 `ManifestValidationError`。

- [x] **Step 1: 写失败测试**

```python
def test_manifest_requires_group_consistency(tmp_path):
    rows = [
        make_known_row(sample_id="s1", group_id="g1", claimed_herb_name="当归"),
        make_known_row(sample_id="s2", group_id="g1", claimed_herb_name="黄芪"),
    ]
    with pytest.raises(ManifestValidationError, match="group_id=g1"):
        validate_known_manifest(rows)


def test_verified_species_requires_nonempty_label():
    row = make_known_row(actual_species=(), species_annotation_status="verified")
    with pytest.raises(ManifestValidationError):
        validate_known_manifest([row])


def test_unknown_and_known_content_hashes_must_not_overlap():
    known = [make_validated_record(sha256="same")]
    unknown = [make_unknown_record(sha256="same", dataset_role="calibration")]
    with pytest.raises(ManifestValidationError, match="内容哈希"):
        validate_global_isolation(known, unknown)
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_manifest.py -q`

Expected: FAIL。

- [x] **Step 3: 实现解析与校验**

本阶段只做切分前校验：安全根目录与路径越界、必填列、枚举、NFC、控制字符、唯一图片路径、解码、像素限制、重复内容/近重复、`sample_id -> group_id` 唯一映射、同组声称标签与真伪标签一致、`verified` 物种非空、每个“声称药材 × 真伪”分层总 group 数不少于三个，并验证全部 known 与全部 unknown 的 group、sample、SHA-256 和感知哈希不交叉。不得引用 train/val/test、不得在本任务生成 taxonomy。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_manifest.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/data/__init__.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/data/manifest.py Chinese_Medicine_Authentication/tests/unit/test_manifest.py Chinese_Medicine_Authentication/tests/fixtures/manifest_minimal.csv Chinese_Medicine_Authentication/tests/fixtures/unknown_manifest_minimal.csv Chinese_Medicine_Authentication/tests/fixtures/validation_report_expected.json
git commit -m "feat: 定义三任务数据清单契约"
```

---

### Task 4: 实现确定性的 group-level 分层切分

**Files:**
- Create: `项目/Files/herb_authentication/data/splitting.py`
- Create: `项目/Files/herb_authentication/data/labels.py`
- Test: `tests/unit/test_splitting.py`
- Test: `tests/unit/test_labels.py`

**Interfaces:**
- `create_group_splits(rows: Sequence[ValidatedImageRecord], ratios=(0.7, 0.2, 0.1), seed=42) -> SplitAssignment`。
- 分层键为 `(claimed_herb_name, authenticity)`；分配单位为 `group_id`；同一 `sample_id`、`group_id`、图片 SHA-256 或感知近重复簇不得跨切分。
- 输入行顺序不影响结果；每个满足最小组数的层都必须出现在 train/val/test。

- `apply_split(rows, assignment) -> tuple[PartitionedImageRecord, ...]` 生成唯一带 `partition=train/val/test` 的记录类型；Task 3 的 pre-split 类型不含 partition。
- `build_label_space(partitioned_rows)` 只从 train 的 claimed 标签和 `verified` 物种冻结 taxonomy；val/test 未见 claimed class 失败，未见物种进入 `unknown_species_labels`。
- [x] **Step 1: 写失败测试**

```python
def test_split_has_no_group_sample_or_hash_leakage(rows):
    assignment = create_group_splits(rows, (0.7, 0.2, 0.1), seed=7)
    assert_no_intersection(assignment, key=lambda row: row.group_id)
    assert_no_intersection(assignment, key=lambda row: row.sample_id)
    assert_no_intersection(assignment, key=lambda row: row.sha256)


def test_split_is_independent_of_input_order(rows):
    first = create_group_splits(rows, (0.7, 0.2, 0.1), seed=7)
    second = create_group_splits(list(reversed(rows)), (0.7, 0.2, 0.1), seed=7)
    assert first.by_group == second.by_group

def test_validation_claimed_class_must_exist_in_train(partitioned_rows):
    rows = add_unseen_validation_claim(partitioned_rows, "黄芪")
    with pytest.raises(ManifestValidationError, match="训练未见"):
        build_label_space(rows)
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_splitting.py tests/unit/test_labels.py -q`

Expected: FAIL。

- [x] **Step 3: 实现稳定切分**

先按分层键聚合 group，以固定 seed 对规范化且稳定排序的 group ID 洗牌，再用最大剩余法按比例分配并保证三个分区各至少一个 group。切分后验证 known train/val/test 之间以及它们与 unknown calibration/test 之间的 group、sample、SHA-256、感知近重复均不交叉，并验证 val/test claimed class 训练覆盖；导入 locked split 时同样执行全部校验。随后从 train 冻结 taxonomy，输出 `train.csv`、`val.csv`、`test.csv`、label space 与包含算法版本、seed、ratios、group 映射和输入数据指纹的 `splits.json`。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_splitting.py tests/unit/test_labels.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/data/splitting.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/data/labels.py Chinese_Medicine_Authentication/tests/unit/test_splitting.py Chinese_Medicine_Authentication/tests/unit/test_labels.py
git commit -m "feat: 增加防泄漏分组切分"
```

---

### Task 5: 构建独立 transform 的三任务 Dataset

**Files:**
- Create: `项目/Files/herb_authentication/data/transforms.py`
- Create: `项目/Files/herb_authentication/data/datasets.py`
- Test: `tests/unit/test_datasets.py`

**Interfaces:**
- `HerbImageDataset(rows, label_space, transform, image_loader)` 返回 `image`、三个 target、`actual_species_mask` 和实体元数据。
- train/val/test 各自持有 transform，不共享或修改底层 Dataset 的 `.transform`。
- 物种 target 为固定长度 multi-hot；非 `verified` 行 mask 为 `False`。
- 验证/测试出现训练词表外的 verified 物种时返回 `unknown_species_labels` 并从闭集物种损失/指标分母排除，另计 unknown-species 指标，不能把它编码成全零负样本。

- [x] **Step 1: 写回归测试**

```python
def test_validation_transform_cannot_mutate_training_transform(dataset_factory):
    train = dataset_factory(transform=lambda image: "train")
    val = dataset_factory(transform=lambda image: "val")
    assert train[0]["image"] == "train"
    assert val[0]["image"] == "val"
    assert train[0]["image"] == "train"


def test_unverified_species_has_false_mask(dataset_factory):
    item = dataset_factory(species_annotation_status="unverified")[0]
    assert item["actual_species_mask"].item() is False
    assert item["actual_species_target"].sum().item() == 0
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_datasets.py -q`

Expected: FAIL；旧实现的共享 transform 回归测试也应失败。

- [x] **Step 3: 实现 Dataset 与 transform factory**

图片以 `Image.open(...).convert("RGB")` 读取；训练增强与验证/测试预处理由独立 factory 创建。DataLoader generator 使用显式 seed，worker 初始化函数派生 Python/NumPy seed。图片读取错误包装为带路径的 `InputImageError`。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_datasets.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/data/transforms.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/data/datasets.py Chinese_Medicine_Authentication/tests/unit/test_datasets.py
git commit -m "feat: 构建隔离变换的多任务数据集"
```

---

### Task 6: 实现可注入骨干的三任务模型与冻结状态机

**Files:**
- Create: `项目/Files/herb_authentication/models/__init__.py`
- Create: `项目/Files/herb_authentication/models/multitask.py`
- Test: `tests/unit/test_multitask_model.py`

**Interfaces:**
- `MultiTaskHerbModel(encoder, feature_dim, num_claimed, num_species)`。
- `forward(images) -> MultiTaskLogits(claimed_herb, actual_species, authenticity)`。
- `set_encoder_trainable(bool)` 必须真实切换 encoder 参数；训练阶段状态可序列化。

- [x] **Step 1: 写失败测试**

```python
def test_multitask_output_shapes(tiny_model, torch):
    output = tiny_model(torch.randn(2, 3, 8, 8))
    assert output.claimed_herb.shape == (2, 3)
    assert output.actual_species.shape == (2, 5)
    assert output.authenticity.shape == (2, 2)


def test_freeze_then_unfreeze_changes_encoder_requires_grad(tiny_model):
    tiny_model.set_encoder_trainable(False)
    assert not any(p.requires_grad for p in tiny_model.encoder.parameters())
    tiny_model.set_encoder_trainable(True)
    assert all(p.requires_grad for p in tiny_model.encoder.parameters())
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_multitask_model.py -q`

Expected: FAIL。

- [x] **Step 3: 实现共享骨干与三个头**

生产 factory 支持配置指定 torchvision 骨干并显式控制是否使用本地已有权重；单元测试注入 `TinyEncoder`。禁止在模块导入时下载权重。冻结期只训练三个头，解冻时重建或补充优化器参数组并保留可恢复的阶段标记。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_multitask_model.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/models Chinese_Medicine_Authentication/tests/unit/test_multitask_model.py
git commit -m "feat: 实现三任务共享骨干模型"
```

---

### Task 7: 实现掩码损失、实体聚合指标与模型选择分数

**Files:**
- Create: `项目/Files/herb_authentication/training/__init__.py`
- Create: `项目/Files/herb_authentication/training/losses.py`
- Create: `项目/Files/herb_authentication/training/metrics.py`
- Test: `tests/unit/test_losses.py`
- Test: `tests/unit/test_metrics.py`

**Interfaces:**
- `compute_multitask_loss(logits, targets, masks, weights) -> LossBreakdown`。
- 声称药材和真伪使用 cross entropy；实际物种使用 masked BCE-with-logits。
- 指标先按 `sample_id` 聚合图片概率，再计算 task macro-F1；总分为有效任务 macro-F1 的调和平均。

- [x] **Step 1: 写关键掩码测试**

```python
def test_unverified_species_rows_do_not_change_loss(torch):
    logits_a = torch.tensor([[0.0, 0.0], [100.0, -100.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, 0.0], [-100.0, 100.0]], requires_grad=True)
    target = torch.zeros_like(logits_a)
    mask = torch.tensor([True, False])
    assert masked_species_loss(logits_a, target, mask) == masked_species_loss(logits_b, target, mask)


def test_all_unverified_species_returns_differentiable_zero(torch):
    logits = torch.randn(2, 3, requires_grad=True)
    loss = masked_species_loss(logits, torch.zeros_like(logits), torch.tensor([False, False]))
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_losses.py tests/unit/test_metrics.py -q`

Expected: FAIL。

- [x] **Step 3: 实现损失与指标**

全未核验 batch 的物种损失使用 `logits.sum() * 0.0` 保留计算图。聚合使用同一 sample 图片概率的算术平均；group bootstrap 以 `group_id` 为抽样单位。物种监督覆盖不足配置阈值时只报告，不参与调和平均。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_losses.py tests/unit/test_metrics.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/training/losses.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/training/metrics.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/training/__init__.py Chinese_Medicine_Authentication/tests/unit/test_losses.py Chinese_Medicine_Authentication/tests/unit/test_metrics.py
git commit -m "feat: 增加掩码损失与实体级指标"
```

---

### Task 8: 建立版本化 checkpoint 与确定性恢复

**Files:**
- Create: `项目/Files/herb_authentication/training/checkpoints.py`
- Test: `tests/unit/test_checkpoints.py`
- Test: `tests/integration/test_resume_training.py`

**Interfaces:**
- `CheckpointMetadata` 包含设计规范第 10 节列出的全部状态、数据 provenance 与数据指纹；`calibration` 预留稳定的纯 primitive/tensor schema，Task 10 只填值不改变结构。
- `save_checkpoint_atomic(path, state, protected_paths)` 先规范化绝对路径，拒绝两个旧权重路径，再写同目录临时文件、校验后原子替换。
- `load_checkpoint(path, expected_contract)` 使用可信文件和 weights-only 安全路径；不兼容抛 `CheckpointCompatibilityError`。
- checkpoint state 只能由 `torch.load(weights_only=True)` 支持的 primitive、容器和 tensor 组成，不直接序列化 dataclass 或任意 NumPy 对象。

- [x] **Step 1: 写 schema 和恢复失败测试**

```python
@pytest.mark.parametrize("legacy_name", ["best_model.pth", "epoch_001.pth"])
def test_save_rejects_both_legacy_paths(project_paths, valid_state, legacy_name):
    legacy_path = project_paths.models_dir / legacy_name
    before = sha256_file(legacy_path)
    with pytest.raises(CheckpointCompatibilityError, match="旧模型"):
        save_checkpoint_atomic(legacy_path, valid_state, project_paths.legacy_model_paths)
    assert sha256_file(legacy_path) == before


def test_atomic_write_failure_preserves_previous_checkpoint(checkpoint_harness):
    before = checkpoint_harness.write_valid()
    checkpoint_harness.fail_next_replace()
    with pytest.raises(OSError):
        checkpoint_harness.write_updated()
    assert checkpoint_harness.load() == before


def test_old_single_head_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "old.pth"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(CheckpointCompatibilityError, match="schema_version"):
        load_checkpoint(path, expected_contract())


def test_resume_matches_continuous_training(resume_harness):
    continuous = resume_harness(steps_before_save=4, steps_after_resume=0)
    resumed = resume_harness(steps_before_save=2, steps_after_resume=2)
    assert_state_tree_equal(continuous, resumed)
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_checkpoints.py tests/integration/test_resume_training.py -q`

Expected: FAIL。

- [x] **Step 3: 实现 checkpoint 契约**

保存模型、优化器、scheduler、scaler、epoch/step、最佳指标、早停计数、冻结阶段、Python/NumPy/Torch CPU/CUDA RNG、DataLoader generator、类别、预处理、校准 schema、阈值、配置快照、实际编码、provenance、依赖版本、git commit、manifest/split/image 指纹。恢复 harness 每步显式消耗 Python/NumPy/Torch RNG，并用 dropout 或等价随机分支验证恢复；固定 CPU、同一依赖、`num_workers=0` 且只在 step 边界保存。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_checkpoints.py tests/integration/test_resume_training.py -q`

Expected: PASS，连续与恢复后的模型、优化器、scheduler、RNG 和冻结阶段逐项一致。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/training/checkpoints.py Chinese_Medicine_Authentication/tests/unit/test_checkpoints.py Chinese_Medicine_Authentication/tests/integration/test_resume_training.py
git commit -m "feat: 增加版本化检查点与确定性恢复"
```

---

### Task 9: 实现 Trainer、冻结调度和训练 CLI

**Files:**
- Create: `项目/Files/herb_authentication/training/trainer.py`
- Rewrite: `项目/Files/train.py`
- Test: `tests/unit/test_trainer.py`
- Test: `tests/integration/test_train_cli.py`

**Interfaces:**
- `Trainer.fit(train_loader, val_loader, resume_from=None) -> TrainingSummary`。
- `train.py` 只负责标准库预检、延迟导入、参数解析、组装依赖和退出码。
- 默认输出新 checkpoint，拒绝写入旧模型路径。

- [x] **Step 1: 写冻结时序与 CLI 失败测试**

```python
def test_encoder_is_frozen_before_unfreeze_epoch(trainer_factory):
    trainer = trainer_factory(unfreeze_epoch=2)
    trainer.run_epoch(0)
    assert trainer.freeze_stage == "heads_only"
    assert not any(p.requires_grad for p in trainer.model.encoder.parameters())
    trainer.run_epoch(2)
    assert trainer.freeze_stage == "full_model"
    assert all(p.requires_grad for p in trainer.model.encoder.parameters())


def test_train_cli_refuses_legacy_output_path(cli_runner):
    result = cli_runner("项目/Files/train.py", "--output", "项目/Models/best_model.pth")
    assert result.returncode == 2
    assert "不允许覆盖旧模型" in result.stderr
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_trainer.py tests/integration/test_train_cli.py -q`

Expected: FAIL；旧代码的冻结条件回归测试应失败。

- [x] **Step 3: 实现训练循环**

每个 epoch 记录三个任务 loss、实体级指标、学习率、冻结阶段和耗时。早停及最佳模型依据任务调和平均分；实际物种监督覆盖不足时排除该任务。AMP 仅在支持设备显式启用。所有随机种子、deterministic 配置和非确定性警告写入训练摘要。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_trainer.py tests/integration/test_train_cli.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/training/trainer.py Chinese_Medicine_Authentication/项目/Files/train.py Chinese_Medicine_Authentication/tests/unit/test_trainer.py Chinese_Medicine_Authentication/tests/integration/test_train_cli.py
git commit -m "feat: 重写可恢复多任务训练流程"
```

---

### Task 10: 实现概率校准与确定性阈值选择

**Files:**
- Create: `项目/Files/herb_authentication/inference/__init__.py`
- Create: `项目/Files/herb_authentication/inference/calibration.py`
- Create: `项目/Files/herb_authentication/inference/policy.py`
- Create: `项目/Files/calibrate.py`
- Test: `tests/unit/test_calibration.py`
- Test: `tests/unit/test_rejection_policy.py`
- Test: `tests/integration/test_calibrate_cli.py`

**Interfaces:**
- `fit_calibrators(known_validation, unknown_calibration) -> CalibrationBundle`。
- `select_thresholds(..., max_unknown_accept_rate=0.05, min_known_selective_accuracy=0.95)`。
- `apply_rejection_policy(scores, bundle) -> RejectionDecision`。
- `calibrate.py --checkpoint INPUT --known-val VAL.csv --unknown UNKNOWN.csv --output OUTPUT` 只接受 locked known validation 与 `dataset_role=calibration`，默认新建 `best_multitask_model.calibrated.pth`，绝不原地修改输入。

- [x] **Step 1: 写 fail-closed 和确定性测试**

```python
def test_missing_calibration_fails_closed():
    decision = apply_rejection_policy(high_confidence_scores(), CalibrationBundle.uncalibrated())
    assert decision.rejected is True
    assert decision.reason_codes == ("model_uncalibrated",)



@pytest.mark.parametrize("low_head", ["claimed_herb", "actual_species", "authenticity"])
def test_any_low_confidence_head_rejects_entire_prediction(calibrated_bundle, low_head):
    decision = apply_rejection_policy(scores_with_low_head(low_head), calibrated_bundle)
    assert decision.rejected is True


def test_no_feasible_threshold_keeps_model_uncalibrated(impossible_calibration_fixture):
    bundle = fit_calibrators(**impossible_calibration_fixture)
    assert bundle.is_calibrated is False


def test_calibrate_cli_rejects_test_role_and_preserves_input(cli_runner, checkpoint, unknown_test):
    before = sha256_file(checkpoint)
    result = cli_runner("项目/Files/calibrate.py", "--checkpoint", checkpoint, "--unknown", unknown_test)
    assert result.returncode == 2
    assert "test 数据禁止用于校准" in result.stderr
    assert sha256_file(checkpoint) == before


def test_actual_species_empty_rejects_entire_prediction(calibrated_bundle):
    scores = high_confidence_scores(actual_species=())
    decision = apply_rejection_policy(scores, calibrated_bundle)
    assert decision.rejected is True
    assert "actual_species_unknown" in decision.reason_codes




def test_successful_calibration_roundtrips_without_mutating_sources(
    cli_runner, checkpoint, known_val, unknown_calibration, calibrated_output, project_paths
):
    protected_before = hash_paths((checkpoint, *project_paths.legacy_model_paths))
    result = cli_runner(
        "项目/Files/calibrate.py", "--checkpoint", checkpoint,
        "--known-val", known_val, "--unknown", unknown_calibration,
        "--output", calibrated_output,
    )
    loaded = load_checkpoint(calibrated_output, expected_contract())
    assert result.returncode == 0
    assert loaded["calibration"]["is_calibrated"] is True
    assert loaded["calibration"]["thresholds"] == expected_thresholds()
    assert hash_paths((checkpoint, *project_paths.legacy_model_paths)) == protected_before


def test_species_partial_match_is_not_selectively_correct():
    assert species_exact_match({"当归"}, {"当归", "欧当归"}) is False


def test_threshold_tie_break_is_deterministic(calibration_fixture):
    first = select_thresholds(**calibration_fixture)
    second = select_thresholds(**calibration_fixture)
    assert first == second
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_calibration.py tests/unit/test_rejection_policy.py tests/integration/test_calibrate_cli.py -q`

Expected: FAIL。

- [x] **Step 3: 实现校准与阈值搜索**

`calibrate.py` 先验证输入 checkpoint 与数据指纹，只用 locked known validation 拟合各头温度或同等可序列化校准器，unknown calibration 只参与拒识阈值选择，任何 test role 立即失败。候选值来自有序唯一置信度及 `0/1` 边界。阈值优化只使用三个标签均可评估且物种均在训练词表内的 known validation 样本：claimed/authenticity 以校准概率 argmax 命中为正确；actual species 以全局 species threshold 二值化后的 taxonomy 内预测集合与 verified 真值集合 exact-set match 为正确；只有三项同时正确才计 selective correct。known selective accuracy 的分母是被三头共同接受的上述样本数，coverage 的分母是全部上述可评估样本数。过滤不满足 unknown accept rate 与 known selective accuracy 约束的组合，排序键固定为 `(-coverage, -selective_accuracy, unknown_accept_rate, -claimed_threshold, -species_threshold, -auth_threshold)`。无可行解返回 `is_calibrated=false`；有可行解把 primitive/tensor `CalibrationBundle` 原子写入新 checkpoint，输入 checkpoint 和两个旧模型保持不变。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_calibration.py tests/unit/test_rejection_policy.py tests/integration/test_calibrate_cli.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/inference/calibration.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/inference/policy.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/inference/__init__.py Chinese_Medicine_Authentication/项目/Files/calibrate.py Chinese_Medicine_Authentication/tests/unit/test_calibration.py Chinese_Medicine_Authentication/tests/unit/test_rejection_policy.py Chinese_Medicine_Authentication/tests/integration/test_calibrate_cli.py
git commit -m "feat: 增加校准与安全拒识策略"
```

---

### Task 11: 实现版本化返回对象、精确中文文案和推理服务

**Files:**
- Create: `项目/Files/herb_authentication/inference/results.py`
- Create: `项目/Files/herb_authentication/inference/formatter.py`
- Create: `项目/Files/herb_authentication/inference/service.py`
- Test: `tests/unit/test_result_formatter.py`
- Test: `tests/integration/test_inference_service.py`

**Interfaces:**
- `InferenceService.from_checkpoint(path, device="auto")`。
- `predict_image(image: Path | bytes) -> PredictionResult`。
- `PredictionResult.to_dict()` 提供未来 Web 可直接 JSON 序列化的 schema v1；精确 `message` 与固定 `disclaimer="模型结果不能替代专业鉴定。"` 分字段返回，Web/CLI 不自行拼接业务结论。

- [x] **Step 1: 写精确文案测试**

```python
def test_authentic_message_contains_only_verdict_and_confidence(formatter):
    result = accepted_result(authenticity="authentic", confidence=0.9234)
    assert formatter.format(result) == "此药品为正品（置信度：92.3%）。"


def test_counterfeit_message_names_claimed_and_actual(formatter):
    result = accepted_result(
        authenticity="counterfeit",
        confidence=0.881,
        claimed="当归",
        species=("欧当归",),
    )
    assert formatter.format(result) == "此药品为伪品（置信度：88.1%），声称为‘当归’，实际应为‘欧当归’。"


def test_unknown_actual_species_never_leaks_verdict_or_percentage(formatter):
    message = formatter.format(rejected_result("actual_species_unknown"))
    assert message == "无法判断该药品的实际药材及真伪，请人工复核。"
    assert "正品" not in message and "伪品" not in message and "%" not in message
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/unit/test_result_formatter.py tests/integration/test_inference_service.py -q`

Expected: FAIL。

- [x] **Step 3: 实现服务边界**

服务初始化一次加载并验证 checkpoint、类别和预处理；单张推理不重新构建模型。服务限制字节大小、像素数并嗅探实际可解码格式，损坏图片抛 `InputImageError`；冻结的 `Path | bytes` 接口没有声明 MIME 参数，因此上传方 MIME 只由未来 Web 适配层验证。工程异常绝不转换为拒识。返回对象保留三个结构化任务结果、`is_calibrated`、`rejected`、reason code、`message` 和 `disclaimer`。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/unit/test_result_formatter.py tests/integration/test_inference_service.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/herb_authentication/inference/results.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/inference/formatter.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/inference/service.py Chinese_Medicine_Authentication/tests/unit/test_result_formatter.py Chinese_Medicine_Authentication/tests/integration/test_inference_service.py
git commit -m "feat: 提供稳定推理接口与中文结论"
```

---

### Task 12: 重写评估与预测 CLI，分别输出机器结果和用户结论

**Files:**
- Rewrite: `项目/Files/evaluate.py`
- Rewrite: `项目/Files/predict.py`
- Create: `项目/Files/herb_authentication/evaluation.py`
- Test: `tests/integration/test_evaluate_cli.py`
- Test: `tests/integration/test_predict_cli.py`

**Interfaces:**
- `evaluate.py` 读取锁定 split，输出 UTF-8 JSON、UTF-8-SIG CSV 与 known/unknown 指标，不修改阈值。
- `predict.py IMAGE --checkpoint PATH [--json] [--allow-uncalibrated]`。
- 默认预测输出人类可读 `message`；`--json` 输出完整 schema v1，供未来 Web 层复用。

- [x] **Step 1: 写 CLI 失败测试**

```python
def test_predict_cli_prints_only_positive_message_by_default(cli_runner, calibrated_checkpoint, image):
    result = cli_runner("项目/Files/predict.py", image, "--checkpoint", calibrated_checkpoint)
    assert result.stdout.splitlines() == ["此药品为正品（置信度：92.3%）。", "模型结果不能替代专业鉴定。"]


def test_evaluate_marks_synthetic_metrics_as_non_official(cli_runner, synthetic_config):
    result = cli_runner("项目/Files/evaluate.py", "--config", synthetic_config, "--json")
    report = json.loads(result.stdout)
    assert report["data_provenance"] == "synthetic"
    assert report["official_metrics"] is False


def test_allow_uncalibrated_requires_json(cli_runner, image, checkpoint):
    result = cli_runner("项目/Files/predict.py", image, "--checkpoint", checkpoint, "--allow-uncalibrated")
    assert result.returncode == 2
    assert "--json" in result.stderr
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/integration/test_evaluate_cli.py tests/integration/test_predict_cli.py -q`

Expected: FAIL。

- [x] **Step 3: 实现薄 CLI 与评估器**

所有入口先调用 bootstrap preflight，再延迟导入重量级模块。预测中的 `--allow-uncalibrated` 必须与 `--json` 同用，只输出带 `rejected=true` 的开发结构化结果，文案不得出现正品、伪品或百分比。评估报告明确区分 image/sample/group 统计口径，并记录数据指纹、checkpoint 指纹、校准状态和 provenance；只有 `provenance=real` 且 locked test 指纹与 checkpoint 契约一致时 `official_metrics=true`，默认 `unknown` 与 `synthetic` 只输出诊断指标。evaluate 永不拟合校准器或修改 checkpoint。

- [x] **Step 4: 验证 GREEN**

Run: `.venv\python.exe -m pytest tests/integration/test_evaluate_cli.py tests/integration/test_predict_cli.py -q`

Expected: PASS。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/项目/Files/evaluate.py Chinese_Medicine_Authentication/项目/Files/predict.py Chinese_Medicine_Authentication/项目/Files/herb_authentication/evaluation.py Chinese_Medicine_Authentication/tests/integration/test_evaluate_cli.py Chinese_Medicine_Authentication/tests/integration/test_predict_cli.py
git commit -m "feat: 重写评估与预测入口"
```

---

### Task 13: 完成端到端验收、迁移说明与 Web 接口契约

**Files:**
- Create: `tests/integration/test_end_to_end_smoke.py`
- Create: `docs/api/inference-service-v1.md`
- Create: `docs/data/manifest-v1.md`
- Modify: `使用指南.md`
- Modify: `.gitignore`

**Interfaces:**
- 文档固定未来 Web 层调用 `InferenceService.predict_image(bytes)`，当前版本不创建 HTTP 服务和浏览器页面。
- 数据文档给出真假、混入、缺失物种、unknown calibration/test 的合法示例与采集要求。

- [x] **Step 1: 写端到端 smoke test**

```python
def test_tiny_train_save_load_predict_pipeline(tiny_project):
    summary = tiny_project.train_one_epoch()
    loaded = tiny_project.load_checkpoint(summary.last_checkpoint)
    result = loaded.predict_image(tiny_project.image_bytes)
    assert result.schema_version == 1
    assert result.rejected is True
    assert result.message == "模型尚未完成真实数据校准，请人工复核。"


def test_calibrated_counterfeit_pipeline_outputs_exact_message(tiny_project):
    trained = tiny_project.train_one_epoch()
    calibrated = tiny_project.calibrate(
        trained.last_checkpoint,
        known_validation=tiny_project.known_validation,
        unknown_calibration=tiny_project.unknown_calibration,
    )
    service = tiny_project.load_checkpoint(calibrated.output_checkpoint)
    result = service.predict_image(tiny_project.counterfeit_image_bytes)
    assert result.is_calibrated is True
    assert result.rejected is False
    assert result.message == "此药品为伪品（置信度：88.1%），声称为‘当归’，实际应为‘欧当归’。"
```

- [x] **Step 2: 验证 RED**

Run: `.venv\python.exe -m pytest tests/integration/test_end_to_end_smoke.py -q`

Expected: FAIL，文档和完整组装尚未完成。

- [x] **Step 3: 编写文档并完成组装**


`使用指南.md` 分别提供：环境诊断、manifest 校验、训练、恢复训练、校准、评估、单图预测命令。数据要求明确：每个 `group_id` 只能属于一个切分；声称药材来自包装/送检信息；实际物种必须来自可追溯专家或实验鉴定；真伪标签需有判定依据；同一实体多角度图片共享 `sample_id`；保留来源、设备、批次和时间。known 与 unknown 各分区的 group/sample/内容/近重复不得重叠；只有 `unknown_type=unknown_herb` 的语义类别必须在已知 taxonomy 外，`blurred/occluded` 可来自已知类。API 文档固定字段、异常、线程安全约束；当前服务只验证字节大小、像素和实际解码格式，上传声明的 MIME 由未来 Web 适配层验证。

- [x] **Step 4: 全量验证**

Run: `.venv\python.exe -m pytest -q`

Expected: PASS。

Run: `.venv\python.exe -m ruff check 项目/Files tests`

Expected: PASS。

Run: `.venv\python.exe 项目/Files/doctor.py --json`

Expected: 返回码 `0` 且报告隔离环境依赖兼容。

Run: `Get-FileHash -Algorithm SHA256 项目/Models/best_model.pth,项目/Models/epoch_001.pth`

Expected: 两个 SHA-256 均与实施前记录值一致，确认旧模型内容未修改；Task 8 的底层写入 API 测试同时保护这两个规范化绝对路径。

- [x] **Step 5: Commit**

```bash
git add Chinese_Medicine_Authentication/tests/integration/test_end_to_end_smoke.py Chinese_Medicine_Authentication/docs/api/inference-service-v1.md Chinese_Medicine_Authentication/docs/data/manifest-v1.md Chinese_Medicine_Authentication/使用指南.md Chinese_Medicine_Authentication/.gitignore
git commit -m "docs: 完成数据规范与接口验收说明"
```

---

## Final Review Gate

- [x] 对照设计规范逐条核对：三任务语义、group 防泄漏、多标签掩码、校准阈值、实际物种未知整体拒识、精确中文文案、旧模型隔离、UTF-8/GBK、未来 Web 接口。
- [x] 搜索并清除 `TODO`、`TBD`、`pass`、占位异常和未实现分支。
- [x] 使用独立代码审查子代理检查正确性、可维护性、安全边界和测试缺口；只修复有证据的问题。
- [x] 在干净的新进程中重跑全量 pytest、ruff、doctor 以及 train/calibrate/evaluate/predict 四个业务 CLI 的 `--help`。
- [x] 最终交付分别说明：已完成内容、架构原理、真实数据要求、运行方式、可能问题与限制。
