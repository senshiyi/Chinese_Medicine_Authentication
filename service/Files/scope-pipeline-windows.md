# 范围门控管线（Windows）使用说明

> 范围门控是对“是否为本药材/本数据集范围”的独立决策，与闭集真伪模型解耦。
> 默认（`SCOPE_ENFORCEMENT=true` 且未配置工件）为 **unavailable 门控**：只有
> `scope=accepted` 的图片才给出公开真伪结论，rejected / unavailable 一律 inconclusive，
> **绝不伪造**真伪预测。

## 0. 前置

- 需要带 **torch + sklearn** 的 Python（本仓库验证环境为实验侧 `项目/.venv`
  `..\项目\.venv\python.exe`，Conda Python 3.12，含
  torch 2.6 CPU + sklearn）。
- 只读引用原项目模型，**不复制权重、不覆盖原参数、不在用户案件图片上自动训练**。
- 语法统一：在 **仓库根目录** 执行，用 `-B` 避免在原项目写字节码，并追加
  `var\venv-stage-a\Lib\site-packages` 提供 API 依赖。

```powershell
$env:PYTHONPATH = 'apps\api'
# 以下每个命令都带上：
#   & '..\项目\.venv\python.exe' -B -c "import sys;sys.path.append(r'$PWD\var\venv-stage-a\Lib\site-packages');sys.path.insert(0,'apps/api');from drynaria_api.cli import main;raise SystemExit(main([...]))"
# 下面对应命令以 <FORWARD> 为该前置包装片段。
```

## 1. 清单 CSV 模板

`scope-manifest-template.csv` 给出了示例列与占位路径（模板本身不包含任何真实图片路径，
也不会被自动训练）：

```text
image_path,scope_label,split,group_id,source,consent
```

校验规则（训练前强制）：

- `scope_label` ∈ {`in_scope`, `out_of_scope`}（**绝不从闭集真伪标签或用户上传推断**）。
- `split` ∈ {`train`, `calibration`, `test`}。
- `consent` 必须为真；未授权样本拒绝参与训练。
- `train`/`calibration`/`test` 的 `group_id` 互不重叠（去分组泄漏）。
- 任意两张图内容（SHA-256）不得重复（同图不许被复用标注）。

## 2. 校验清单

```powershell
& '..\项目\.venv\python.exe' -B -c "import sys;sys.path.append(r'$PWD\var\venv-stage-a\Lib\site-packages');sys.path.insert(0,'apps/api');from drynaria_api.cli import main;raise SystemExit(main(['scope-validate-manifest','--manifest','<路径>\你的清单.csv','--image-root','<图片根目录>']))"
```

输出示例：`scope manifest OK` + 各 split 的 `in_scope / out_of_scope` 计数。

## 3. 训练 / 校准 / 评估并写出工件

```powershell
& '..\项目\.venv\python.exe' -B -c "import sys;sys.path.append(r'$PWD\var\venv-stage-a\Lib\site-packages');sys.path.insert(0,'apps/api');from drynaria_api.cli import main;raise SystemExit(main(['scope-train','--manifest','<路径>\你的清单.csv','--image-root','<图片根目录>','--out','<输出>\scope-artifact.json','--checkpoint','..\项目\Models\best_multitask_model.calibrated.pth','--inference-project','..\项目\Files','--max-false-accept','0.05','--max-false-reject','0.05']))"
```

- 只用 `train` 拟合 scaler+logistic regression；只用 `calibration` 选阈值；只在 `test` 评估一次。
- 报告计数、错误率与 95% Wilson 置信区间。
- `approved=true` 仅当 held-out 的两项错误率 **95% Wilson 置信区间上限** 都不超过指定上限；小样本零错误仍可能不达标，工件只作为 **候选**（不部署）。5% 参数只是命令示例，不代表已批准的产品验收标准；样本相关性仍需按独立实物、来源和拍摄分组控制。
- 工件绑定 `checkpoint_sha256` 与 `feature_dim`，全部数值为 JSON（无 pickle）。

## 4. 校验工件

```powershell
& '..\项目\.venv\python.exe' -B -c "import sys;sys.path.append(r'$PWD\var\venv-stage-a\Lib\site-packages');sys.path.insert(0,'apps/api');from drynaria_api.cli import main;raise SystemExit(main(['scope-verify','--artifact','<输出>\scope-artifact.json','--checkpoint','..\项目\Models\best_multitask_model.calibrated.pth']))"
```

绑定不一致、损坏、或 `approved=false` 的工件会被拒绝，运行时门控退化为 unavailable。

## 5. 启用范围门控（服务侧）

在 `.env` 配置：

```text
SCOPE_ENFORCEMENT=true
SCOPE_ARTIFACT_PATH=<绝对路径>\scope-artifact.json
```

未配置工件时默认即为 unavailable 门控（所有案件 inconclusive）。历史（实验闭集）预测仍
保留在库中，但公开 API / 前端在无范围证据时**不会把它当作有效真伪结论呈现**。

## 6. 已知限制（务必阅读）

- 目前**没有**经独立标注且授权的真实 out-of-scope 语料，因此本仓库**不部署**任何真实
  范围工件；默认 unavailable 门控是诚实的保守行为，**不是**学到开放集（OOD）识别。
- 保守拒识（一律 inconclusive）并不等于学会了“非本药材”的识别；该结论只在获得独立
  标注、授权、分组隔离、且通过 held-out 评估的语料后才成立。
- 合成的管线自检仅证明确机械正确（train/calibration/test 阶段划分、校验、JSON 工件、
  绑定、阈值选择），**不代表真实识别准确率**。
