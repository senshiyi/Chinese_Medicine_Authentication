# 推理服务接口契约 v1

当前版本提供稳定的 Python 推理接口，不启动 HTTP 服务，也不包含浏览器上传页面。未来 Web 适配层只负责鉴权、限流、上传 MIME 校验和 JSON 响应；模型加载、图片内容校验、校准、拒识以及中文结论统一由本接口负责。

## 调用方式

```python
from pathlib import Path
from herb_authentication.inference.service import InferenceService

service = InferenceService.from_checkpoint(
    Path("项目/Models/best_multitask_model.calibrated.pth"),
    device="auto",
)
result = service.predict_image(upload_bytes)
payload = result.to_dict()
```

`InferenceService` 应在进程启动时构建一次并复用，不能为每次请求重复加载模型。`predict_image` 使用推理模式，不修改模型和校准参数。若 Web 服务器采用多线程并发，建议为 GPU 推理设置队列或互斥，并通过多进程扩展吞吐量，不依赖底层加速库的隐式线程安全。

## 输入契约

- 接受 `pathlib.Path | bytes`；Web 层使用 `bytes`。
- 核心服务限制压缩后字节数、像素数和估算的 RGB 解码字节数，并使用 Pillow 实际解码内容。
- 文件扩展名和客户端声明的 MIME 不可信。未来 Web 层还必须独立校验 MIME、请求体大小、认证和速率限制。
- 空文件、损坏图片、解压炸弹、超限图片抛出 `InputImageError`；checkpoint 不兼容抛出 `CheckpointCompatibilityError`。工程错误不会伪装成“无法判断”。

## 返回 schema v1

`PredictionResult.to_dict()` 只包含 JSON 原生类型：

```json
{
  "schema_version": 1,
  "model_version": "multitask-v1",
  "is_calibrated": true,
  "message": "此药品为伪品（置信度：88.1%），声称为‘当归’，实际应为‘欧当归’。",
  "disclaimer": "模型结果不能替代专业鉴定。",
  "claimed_herb": {"label": "当归", "confidence": 0.94},
  "actual_species": [{"label": "欧当归", "confidence": 0.91}],
  "authenticity": {"label": "counterfeit", "confidence": 0.881},
  "rejected": false,
  "rejection_reasons": []
}
```

字段只做向后兼容扩展；破坏性修改必须增加 `schema_version`。Web 页面必须直接显示 `message` 与独立的 `disclaimer`，不得自行拼接或推断业务结论。

## 拒识规则

- checkpoint 未完成真实校准时 fail closed，不输出正品、伪品或百分比。
- 声称药材、实际物种、真伪任一任务低于已锁定阈值时拒识。
- 实际物种没有任何标签达到阈值时，返回“无法判断该药品的实际药材及真伪，请人工复核。”，即使真伪头置信度很高也不作判断。
- 只有三个任务均被接受时才输出真伪结论；正品只显示结论和置信度，伪品额外显示声称药材及实际应为的物种。

## Web 适配层预留建议

未来可新增 `/api/v1/predictions`，接收单个 multipart 图片并将其字节传给本服务。适配层还需要实现身份认证、CSRF/CORS 策略、限流、超时、请求 ID、审计日志及隐私数据留存策略；这些能力不属于当前版本。
