# 范围门控（Scope Rejection）实施与验收报告

> 对应计划：`docs/superpowers/plans/2026-09-10-scope-rejection.md`
> 范围：`E:\Desktop\WorkSpace\Chinese_Medicine_Authentication_v2`（阶段 A 服务）。
> 原闭集实验项目（`../Chinese_Medicine_Authentication`）保持只读，未复制/覆盖任何模型权重。

## Codex 独立验收与部署（2026-09-10）

- DSH headless 完成主体实现，进程正常退出。Codex 补充：批准必须满足错误率 95% Wilson 区间上限；严格类别映射校验；公开结果屏蔽无范围证据的历史肯定文案。
- 全量后端独立测试：136 passed、1 skipped（Docker）。随后修改对应的范围测试 41 passed，历史案件 API 测试 5 passed。
- 最终前端：11 tests passed；类型检查与生产构建通过。新增组件测试确认即使返回旧的 99.9% 正品文案，未验证范围时也不展示。
- 真实原模型特征提取冒烟：1280 维，与模型匹配。合成图片仅验证接口，不用于训练或性能声明。
- 原数据清单记录 known=427、unknown=0；未得到额外已标注授权的范围外数据，未训练或部署真实范围工件。
- 本地开发库已用 SQLite backup 备份到 `var/dev-preview.before-scope-20260910.db`；核对 11 张表与原迁移基线后登记版本，升级至 `c9d4e2f1a8b0`，保留案件与历史输出。
- 本地服务已重启。8 个已有案件检查通过：公开结论无效、结果为 inconclusive、无旧的肯定正品文案。网页 `http://127.0.0.1:5174/` 保持运行。
- 限制：当前是缺少校准证据时的保守拒绝，不是已经训练完成的非药材识别。真实模型性能仍待独立标注、授权和分组隔离的数据验证。

下文为 DSH 实施记录，前端沙箱限制由上述独立验证结果补充。

## 结论

已实现并验证范围门控（accepted / rejected / unavailable）的完整工程链：清单校验 → 冻结
encoder 特征 → sklearn 训练/校准/评估 → JSON 工件（绑定 checkpoint SHA-256 与特征维度）
→ 运行时门控 → Worker 结果映射 → API/前端契约。**默认 unavailable 门控生效**：只有
`scope=accepted` 的图片才给出公开真伪结论；`rejected` / `unavailable` 一律 inconclusive，
不伪造真伪预测；历史/实验闭集预测仍保留且不可变，但公开接口不再把它当作有效结论呈现。

当前**没有**经独立标注且授权的真实 out-of-scope 语料，因此**未部署任何真实范围工件**；
范围管线可执行，默认 `SCOPE_ENFORCEMENT=true` 且 `SCOPE_ARTIFACT_PATH` 为空时为
unavailable 门控。保守拒识**不等于**学到开放集（OOD）识别。

## 已实现

### 范围子包 `drynaria_api/scope/`
- `manifest.py`：清单读取与校验（列、`in_scope/out_of_scope`、`train/calibration/test`、
  `source`、`group_id`、授权 consent；分组隔离 + 重复内容（SHA-256）检测）。纯标准库。
- `artifact.py`：JSON 工件（无 pickle）序列化/读取/校验，绑定 checkpoint SHA-256 与
  feature_dim，拒绝损坏/不匹配/未批准（候选）工件。纯标准库。
- `gate.py`：运行时门控 `ScopeDecision{accepted,rejected,unavailable}`，用冻结特征 +
  StandardScaler/LogisticRegression 概率与阈值判定；`build_scope_gate` 绝不抛错、一律
  退化为 unavailable；绑定需能读取当前 checkpoint（不可读即为 unavailable），未批准/不匹配/
  维度不符也一律 unavailable；`scope_fingerprint` 供幂等键使用。纯标准库。
- `features.py`：冻结 encoder 特征提取（复用原项目 `InferenceService` 的共享骨干，只读）。
  仅 Worker/训练进程按需导入（含 torch）。
- `pipeline.py`：只在 train 拟合 scaler+logreg；只在 calibration 选阈值；只在 test 评估一次；
  受操作者给定 max-false-accept / max-false-reject 约束，报告计数、错误率与 95% Wilson 置信
  区间；held-out 通过才 `approved=true`，否则只写候选。仅带 sklearn+torch 进程导入。

### 集成
- `settings.py`：`scope_enforcement=true`（默认）、`scope_artifact_path=""`。
- `worker.py`：`compute_case_outcome(payload, min_confidence, scope_decision)`；门控
  `accepted` 才走闭集结论，否则 inconclusive；`run_analysis_job` 计算并落库 `scope_decision`。
- `cases.py`：API 返回 `scope`/`scope_enforced`/`result_verdict_valid`；`_effective_result_type`
  在无有效范围证据时把展示结果纠正为 inconclusive；分析幂等键加入 `scope` 指纹，避免重放
  pre-gate 结果，历史预测原样保留。
- `main.py`：`/api/v1/status` 报告 `scope_enforcement`/`scope_artifact_available`。
- `models.py` + Alembic `c9d4e2f1a8b0`：`predictions.scope_decision`（JSON，可空）。
- `cli.py`：`scope-validate-manifest`、`scope-train`、`scope-verify` 三个子命令（延迟导入，
  保持隔离）。
- 前端：`api.ts` 类型、`results.ts`（`verdictInfo` + 范围通知文案）、
  `CaseResultView.vue`（无有效范围证据时显示“无法判断”+ 范围通知，不呈现肯定性结论）、
  `CaseListView.vue`（inconclusive / 范围证据缺失文案与色阶）。
- 交付物：`Files/scope-manifest-template.csv`、`Files/scope-pipeline-windows.md`、
  `.env.example`（`SCOPE_ENFORCEMENT`/`SCOPE_ARTIFACT_PATH`）。

## 验证结果

| 项目 | 结果 |
|------|------|
| 后端全量 pytest（`var\venv-stage-a`） | **128 passed, 2 skipped** |
| 范围管线自检（原 `.venv`，含 sklearn+torch） | **8 passed**（合成数据，只证机械正确性） |
| 模型导入隔离 | `torch/sklearn/herb_authentication/scope.features/scope.pipeline` 均**未**被 API 模块导入 |
| 前端 `vue-tsc --noEmit` | **通过** |
| 前端 `vite build` / `vitest` | **未能运行**：esbuild 子服务命名管道 `spawn EPERM`；升级 `danger-full-access` 需审批但无审批通道 |
| CLI 注册与 `scope-verify` 冒烟 | 通过；`scope-validate-manifest` 正确拒绝模板占位路径 |
| Alembic | `c9d4e2f1a8b0` 为 head，链有效 |

> 后端测试需在仓库根目录设 `STORAGE_BACKEND=local`、`S3_ENDPOINT_URL=` 以避开 `.env` 的
> MinIO 端点（模块级 `create_app()` 会在导入时联机）。`test_scope_pipeline` 在无 sklearn 的
> 主测试环境被跳过，单独用原 `.venv`（追加 `var\venv-stage-a\Lib\site-packages`、`-B`）验证。

## 实际限制

- 无真实 OOD 语料 → 不部署范围工件；默认 unavailable 门控诚实保守，**不是**学到 OOD 识别。
- 默认启用门控后，未配置 `SCOPE_ARTIFACT_PATH` 时所有案件结果将显示“无法判断”。
- 前端构建/单元测试在本 DSH 沙箱中因 esbuild 命名管道 `EPERM` 无法运行（vue-tsc 已通过，
  验证 TS 契约正确）；这不是代码问题，是沙箱对 Node 子服务 spawn 的限制。
- 合成的管线自检仅验证阶段划分/校验/JSON 工件/绑定/阈值选择，**不代表真实识别准确率**；
  未跳过任何“最小样本数/错误率上限”作为静默科学验证。

## 未做 / 边界遵守

- 未修改原闭集模型项目；未复制模型权重；未覆盖任何原参数。
- 未安装依赖；未改动服务编排（docker-compose 等）；未提交 git。
- 未在用户案件图片上自动训练；`scope-train` 只读取显式 `--manifest`。
- 未重启任何本地服务（本会话未启动/改动服务；缺失工件行为已通过测试与导入隔离验证）。
