# 骨碎补正伪闭集实验记录（2026-08-09）

本记录对应正式 run `20260825-drynaria-closed-set-v2`，是固定数据集上的闭集真伪二分类实验，不是开放集识别、未知类拒识、药材物种鉴定、医疗建议或质量结论。所有文本文件和 JSON 均使用 UTF-8；项目实际解释器为 `.venv\python.exe`。

## 环境与可追溯范围

- 操作系统：Windows 11 `10.0.26200`；CPU 运行。
- Python `3.12.13`，PyTorch `2.6.0+cpu`，torchvision `0.21.0+cpu`，CUDA 不可用。
- seed `42`；训练时 Git HEAD `8acf46afad6c012240795f8650c02fac8d91cd38`。
- 配置快照 SHA-256：`9e259b43fd46ed8c65a35ce4f5523b7dec544c65fbc990716630db692ad27bc6`；训练/评估配置语义指纹：`6aeba8c88f16bc0a3fcc6dd0b4cb6a5f2ba52ddd6d3e41ba2b9b5cef22d58100`。
- 数据来源为 427 张本地图片；335 张标为正品，92 张标为伪品。427 个 `sample_id` 和 427 个 `group_id` 均独立唯一。缺少批次、来源、采集设备和拍摄会话元数据，不能排除采集捷径。
- 伪品只有真伪标签，实际物种未经检验；模型响应中的 `actual_species_verified=false` 不得解释为物种鉴定结果。

## 数据准备与 v2 切分

正式准备命令：

```powershell
.venv\python.exe 项目\Files\prepare_closed_set.py --config 项目\Files\config_closed_set.yaml
```

最终版本为 `group-stratified-near-duplicate-components-v2`。它对全部 group 做 average-hash 8x8/LANCZOS 全两两比较，以距离 `<=4`（含边界）连边并取传递闭包，再用 `exact-component-dp-v1` 原子分配组件。求解状态为 `optimal`，目标函数 `multi-label-target-proportion-absolute-deviation-v1` 的值为 `424`。

| 分区 | 正品 | 伪品 | 合计 |
|---|---:|---:|---:|
| train | 235 | 64 | 299 |
| validation | 50 | 14 | 64 |
| locked-test | 50 | 14 | 64 |
| 全部 | 335 | 92 | 427 |

审计结果：134 条近重复边；328 个组件，其中 292 个单例、36 个多成员、5 个混合标签；最大组件 36 个 group。独立 oracle 检查全部 90,951 对图片，跨分区距离 `<=4` 为 0，组件跨分区数为 0。

| 产物或指纹 | SHA-256 |
|---|---|
| `metadata.csv` / manifest | `c93b391799ee1787a10a62ac99e54c2919df4f517fbc90aac0a0222040a7412b` |
| `manifest_validation.json` | `cf05fc711c06819cd6b01914f384c6017c48102585fde300a251e0d99f91b093` |
| `train.csv` | `1738c25691bffa14c84b9e2583ed9236019f1ebf57f8c426d34b13736895d4a1` |
| `val.csv` | `870dac09a84715ca275445d8761d9f9cd4de4b7151235cbcb17f98f04d4c4310` |
| `test.csv` | `d4ddb2f89375d88af71b3c9ff9941bb9b846e1f43bf5f286c84ebe6a8c962d62` |
| `splits.json` / split | `931c7705b768311dae4057c1f3e6cd4bcbc2077f137ea1923b9b2c96ed68d495` |
| split input | `6154edc97e3083cb48bad9dc8f4804b20062aae3cda8ae76ae218c4c7be4dbd2` |
| component | `8a923d239aac342ecee2f18968b8597cff7a8513825bdb27f913449bf54ab6ff` |
| component label vectors | `1866f0f9d7443c512dad0eb97260b2add7f3b153ee842ffc8ce322e46be09927` |
| group mapping | `9dc142f90ce11ebab61db5b96f3284f1461cd10137e8ebc6b5ac0db12f997fbe` |

average-hash 可能把视觉相似但并非重复的图片连接起来，也可能漏掉其他类型的重复；组件约束是在该预注册算法下防止切分泄漏的保守措施，不是内容或标签真实性证明。

## 两次训练 run

首次不可变 run `20260809-drynaria-closed-set-v1` 在 epoch 0 前失败：训练装配把 `PartitionedImageRecord` 当成平面 manifest row，访问了不存在的 `authenticity` 字段。失败目录只保留 `config.yaml` 与 `environment.json`，没有候选模型、history、evaluation 或 reservation。根因经真实嵌套记录回归复现，修复提交为 `8acf46afad6c012240795f8650c02fac8d91cd38`；没有复用该 run ID。

成功 run 命令：

```powershell
.venv\python.exe 项目\Files\train.py --config 项目\Files\config_closed_set.yaml --device cpu --run-id 20260825-drynaria-closed-set-v2
```

训练只使用 train，选模只使用 validation。运行 9 个 epoch 后因 patience 耗尽提前停止，耗时约 150.642 秒；最佳 epoch 为 3，选择公式为 `(validation authenticity Macro F1 + Balanced Accuracy) / 2`，最佳分数 `0.9099249061`。真伪类别计数为 235/64，唯一类别平衡机制为 loss class weights `[0.6361702085, 2.3359375]`，`use_weighted_sampler=false`。

| epoch | train total loss | val Macro F1 | val Balanced Accuracy | selection score |
|---:|---:|---:|---:|---:|
| 0 | 0.4597764 | 0.8628571 | 0.8628571 | 0.8628571 |
| 1 | 0.2618274 | 0.8623656 | 0.9300000 | 0.8961828 |
| 2 | 0.1683794 | 0.8623656 | 0.9300000 | 0.8961828 |
| 3 | 0.1576216 | 0.8798498 | 0.9400000 | 0.9099249 |
| 4 | 0.1528479 | 0.8798498 | 0.9400000 | 0.9099249 |
| 5 | 0.1303539 | 0.8798498 | 0.9400000 | 0.9099249 |
| 6 | 0.1069274 | 0.8798498 | 0.9400000 | 0.9099249 |
| 7 | 0.1177425 | 0.8798498 | 0.9400000 | 0.9099249 |
| 8 | 0.0967430 | 0.8798498 | 0.9400000 | 0.9099249 |

源候选 SHA-256 为 `2f65525518d5c8f7fe92b814b1a7be4eddf318ac830efbee221566d04a7ae8b1`；history 全部有限，checkpoint、history 和 selection score 一致。

## validation-only 温度校准

```powershell
.venv\python.exe 项目\Files\calibrate.py --mode closed-set-temperature --checkpoint 项目\Models\experiments\drynaria_closed_set.candidate.pth --known-val 项目\Data\experiments\drynaria_closed_set\val.csv --output 项目\Models\experiments\drynaria_closed_set.calibrated.candidate.pth --data-root 项目\References --device cpu
```

仅使用 validation CSV，耗时约 7.418 秒。温度为 `0.7948504334`；NLL `0.1719675157 → 0.1677257698`，Brier `0.1046956505 → 0.1033135362`，ECE `0.0638519458 → 0.0611805989`。校准候选 SHA-256 为 `e25895c7dce0b1258377561af4a8a5bb73d1e8a6201c627b244d4470a13d1a95`；标志为 `experimental_closed_set=true`、`open_set_calibrated=false`。

## validation 评估

命令只读取 `val.csv`。报告 JSON/CSV SHA-256 分别为 `c3d92ee18ee9a2e761ef9c2d128bbda27e2aa23e2be47cc04840b7517e27316b` 和 `178ce924538e7513a0c716f9bca9e4c35118facb70480c1e7184caa9327aa908`。类别顺序为 `[authentic, counterfeit]`，混淆矩阵为 `[[44, 6], [0, 14]]`。

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| authentic | 1.0000000 | 0.8800000 | 0.9361702 | 50 |
| counterfeit | 0.7000000 | 1.0000000 | 0.8235294 | 14 |

| 指标 | 点估计 | bootstrap 95% lower | median | upper |
|---|---:|---:|---:|---:|
| Macro F1 | 0.8798498 | 0.7971766 | 0.8798498 | 0.9564626 |
| Balanced Accuracy | 0.9400000 | 0.8900000 | 0.9400000 | 0.9800000 |
| authentic recall | 0.8800000 | 0.7800000 | 0.8800000 | 0.9600000 |
| authentic F1 | 0.9361702 | 0.8764045 | 0.9361702 | 0.9795918 |
| counterfeit recall | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 |
| counterfeit F1 | 0.8235294 | 0.7179487 | 0.8235294 | 0.9333333 |
| AUROC | 0.9885714 | 0.9657143 | 0.9900000 | 1.0000000 |
| PR-AUC | 0.9588976 | 0.8882178 | 0.9661428 | 1.0000000 |
| calibrated Brier | 0.1033135 | 0.0379146 | 0.1006515 | 0.1850117 |
| calibrated ECE | 0.0611806 | 0.0386654 | 0.0754759 | 0.1246833 |
| calibrated NLL | 0.1677258 | 0.0727377 | 0.1635822 | 0.2845921 |

未校准 Brier/ECE/NLL 分别为 `0.1046956494`、`0.0638519479`、`0.1719675168`。固定门槛 Macro F1、Balanced Accuracy、counterfeit recall 均为 `0.80`；三个 validation 检查均通过。

## 唯一一次 locked-test

在任何 test logits 前先写入 reservation；reservation SHA-256 为 `e86987afb9846feb2b13ead1ae51c0255bfa7e21b0e91583a88f497f724c7f33`。locked JSON/CSV SHA-256 分别为 `a3494f9a2cb973581f65e1378bb38ec1d5c7bd729526586fd2591ecbacb5fb20` 和 `74d79ffe2fdf7d0d1daf5f9174822d86736ebc87632ff17800bf28979597a718`。没有用 test 指标调参或重跑。类别顺序为 `[authentic, counterfeit]`，混淆矩阵为 `[[46, 4], [2, 12]]`。

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| authentic | 0.9583333 | 0.9200000 | 0.9387755 | 50 |
| counterfeit | 0.7500000 | 0.8571429 | 0.8000000 | 14 |

| 指标 | 点估计 | bootstrap 95% lower | median | upper |
|---|---:|---:|---:|---:|
| Macro F1 | 0.8693878 | 0.7714286 | 0.8693878 | 0.9564626 |
| Balanced Accuracy | 0.8885714 | 0.7814286 | 0.8942857 | 0.9800000 |
| authentic recall | 0.9200000 | 0.8400000 | 0.9200000 | 0.9800000 |
| authentic F1 | 0.9387755 | 0.8888889 | 0.9387755 | 0.9800000 |
| counterfeit recall | 0.8571429 | 0.6428571 | 0.8571429 | 1.0000000 |
| counterfeit F1 | 0.8000000 | 0.6451613 | 0.8000000 | 0.9333333 |
| AUROC | 0.9800000 | 0.9471429 | 0.9828571 | 1.0000000 |
| PR-AUC | 0.9387118 | 0.8543661 | 0.9480468 | 1.0000000 |
| calibrated Brier | 0.1136169 | 0.0366818 | 0.1101590 | 0.2052608 |
| calibrated ECE | 0.0493175 | 0.0206831 | 0.0575569 | 0.1124587 |
| calibrated NLL | 0.1673407 | 0.0630504 | 0.1625397 | 0.2908909 |

未校准 Brier/ECE/NLL 分别为 `0.1076579165`、`0.0583824683`、`0.1589344221`。校准后 ECE 较低，但 NLL 与 Brier 在 locked-test 上略差；这是只在 validation 拟合温度后的独立观察，没有用 locked-test 重新校准或调参。三个固定门禁点估计均不低于 `0.80`，整体通过；置信区间较宽，尤其 14 个伪品样本的 counterfeit recall 下界为 `0.6428571`。

## 晋升与推理语义

预定 smoke 样本是正式 `test.csv` 第一行 `df-auth-0005`，图片 SHA-256 为 `f38278054d600991a63ed6debe9f6ff45e6b430cb08ee7775b0c43a6fabcb3ab`。独立 smoke 的后置脚本误把根级 `probabilities` 当成 `authenticity` 子字段，未保留比较输出；没有重试该独立 smoke。随后正式 `promote.py` 成功（exit 0）：其 preflight 自行对同一图片执行 Path 与 bytes 推理、把两侧转换成完整 JSON 原生字典并做全字典相等比较，只有比较通过才进入临时复制和 `os.replace`。

晋升记录 `promotion.json` SHA-256 为 `fe78f35a5f58bf3ec0d1474152690818d0362caad92e9efca7f17a1f8d8ce013`。`old_sha256=null`，`new_sha256=e25895c7dce0b1258377561af4a8a5bb73d1e8a6201c627b244d4470a13d1a95`，晋升时间 `2026-08-25T13:11:01Z`。正式目标 `项目/Models/best_multitask_model.calibrated.pth` 与校准候选哈希一致并可安全加载；旧 preview `项目/Models/best_model.pth` 仍为 `d67fb4d6268e386d191ae3374f4c5294f8a9854b916301d7bba5dde9b64f2606`。Git 跟踪的 `.pth` 数为 0。

单次预测中的 `probabilities.authentic` / `probabilities.counterfeit` 是模型对该输入在这个闭集二分类内的归一化输出，二者和为 1；它不是实验集准确率，也不表示诊断确定性。上表的 Macro F1、Balanced Accuracy、recall 等才是固定 validation/test 样本集合上的实验指标。二者不可互换。

## 限制与风险

- 这是骨碎补正品/伪品闭集实验，没有 unknown calibration/test，也没有未知药材拒识能力；`open_set_calibrated=false`。
- counterfeit 标签不提供经检验的实际物种，不能输出或暗示伪品物种鉴定。
- 缺少批次、来源、设备、拍摄会话元数据；即使按 group 和 pHash component 隔离，也不能排除所有数据采集捷径。
- average-hash 阈值与组件约束只降低一种可测的近重复泄漏风险，不证明图片独立同分布，也不保证现实部署泛化。
- 样本规模有限，bootstrap 区间体现了明显不确定性。固定 test 只评估一次，结果未用于选择、校准或调参。
- 本实验不是医疗建议、药典鉴定、产品放行或药材质量结论。任何实际使用都需要独立的来源、批次、成分和专业人工验证。