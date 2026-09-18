# 骨碎补在线图像鉴伪平台 V2 设计规格

**日期：** 2026-08-28
**状态：** 已批准
**适用项目：** `Chinese_Medicine_Authentication`

## 1. 背景与决策

项目已有一套骨碎补正品/伪品闭集实验系统，包括数据清单、近重复组件约束切分、EfficientNet-B0 训练、validation 温度校准、一次性 locked test、模型晋升和 Python `InferenceService`。当前没有 HTTP 服务、浏览器页面或真实手机开放集能力。

现有原始数据共 427 张统一白底 JPEG：正品 335 张、伪品 92 张。全部为 1920 x 1080 RGB 图片，无 SHA-256 完全重复，但存在 134 条 average-hash 近重复边、36 个多成员组件和 5 个混合标签组件。数据没有可验证的批次、设备、拍摄会话和伪品实际物种信息。完整审计见 `docs/data/drynaria-raw-eda-2026-08-28.md`。

已比较三条路线：

1. 直接给闭集模型增加网页；交付快，但容易把白底实验结果误解为真实手机鉴伪能力。
2. 等真实数据充分后重新建设；科学边界清晰，但短期没有在线成果。
3. 双轨渐进；先发布明确标注实验范围的在线版，同时建立专家审核和真实数据闭环，再以独立手机数据门禁升级为正式版本。

本设计采用路线 3。

## 2. 目标与非目标

### 2.1 目标

- 为用户提供手机浏览器中的骨碎补拍摄、上传、查询和专家复核流程。
- 复用现有闭集模型，交付可审计的阶段 A 实验性在线服务。
- 明确区分图片质量失败、无法判断、疑似伪品、系统失败和专家结论。
- 建立整根、局部纹理和断面多视角数据采集及专家标注闭环。
- 为具体冒充品、混淆品、炮制质量和未知样本拒识建立可扩展标签体系。
- 只有通过来源、批次、设备独立的手机 locked test 后，才允许晋升为正式鉴伪版本。

### 2.2 非目标

- 阶段 A 不识别伪品实际物种。
- 阶段 A 不使用断面或局部图参与真伪融合判断。
- 阶段 A 不证明未知药材拒识、炮制质量或真实手机场景准确率。
- 系统不替代药典鉴定、显微、理化、分子检测或专业人员结论。
- 本设计不覆盖医疗建议、疗效判断、供应链放行和法律鉴定。

## 3. 三阶段交付边界

### 3.1 阶段 A：实验性在线版

- 支持账户、任务、整根外观图上传、质量检查、异步推理、结果查询和专家复核。
- 复用正式晋升的现有 PyTorch checkpoint 和 `InferenceService`。
- 只用 `view_type=whole` 的主图进行闭集推理。
- 局部和断面图可以上传，但只进入审核和候选数据区。
- 所有模型结果保留 `experimental_closed_set=true` 和 `open_set_calibrated=false`。

### 3.2 阶段 B：真实数据建设版

- 采集整根、局部纹理、断面三类手机图片。
- 增加模糊、曝光、遮挡、主体不完整、复杂背景和非骨碎补样本。
- 专家按实际物种、真伪类型、炮制状态和鉴定依据审核。
- 冻结可追溯数据集版本，禁止未经审核的线上数据自动参与训练。

### 3.3 阶段 C：正式鉴伪版

- 引入主体定位、多任务、多视角融合和开放集拒识。
- 在独立批次、设备、来源的 validation 上选择阈值和校准。
- locked test 一次性评估通过后才允许模型晋升。
- 正式版本仍保留拒绝判断和人工复核路径。

## 4. 总体架构

```text
Vue 3 PWA
    |
Nginx / HTTPS
    |
FastAPI API
    |-- PostgreSQL: 用户、任务、结果、审核、审计
    |-- MinIO: 原图、缩略图、审核附件
    `-- Redis/Celery: 异步任务
             |
        推理 Worker
             |
    质量策略 + InferenceService
             |
       模型与策略结果
```

API 进程不加载模型。推理 Worker 启动时加载并复用一份模型，每个推理设备控制 Worker 数量。阶段 A 继续使用 PyTorch checkpoint；只有实际性能分析证明推理是瓶颈时，才新增 ONNX Runtime 路线，并以黄金图片验证输出兼容性。

## 5. 组件边界

### 5.1 用户 PWA

- 登录、任务历史和删除申请。
- 创建任务并确认图片分析授权。
- 拍摄或选择整根、局部、断面图片。
- 在上传前做格式、大小和基础分辨率提示。
- 显示任务状态、模型结果、适用范围和专家结论。

客户端检查只改善体验，不能替代服务端校验。

### 5.2 API 层

- 身份认证、角色授权、限流、请求 ID 和审计。
- 管理任务状态、图片元数据和幂等分析提交。
- 完成流式大小限制和实际图片解码校验，去除非必要 EXIF 后再保存对象。
- 创建异步任务并交给推理 Worker，不在 API 进程加载模型或重算概率。
- 返回版本化 JSON，不在路由中拼接领域结论。

### 5.3 图片质量与策略层

- 校验真实解码、文件大小、像素数和支持格式。
- 检查模糊、曝光、主体占比和明显裁切。
- 阶段 A 采用可解释规则和固定阈值，阈值由 validation 数据确定。
- 业务拒绝返回 `needs_retake` 或 `inconclusive`；工程异常返回 `failed`。

### 5.4 推理 Worker

- 从 MinIO 读取授权图片，在受控内存边界内解码。
- 只把整根主图传给阶段 A `InferenceService`。
- 记录模型、checkpoint、数据、切分、校准和策略版本。
- 使用幂等任务键，重试不能重复创建预测记录。

### 5.5 专家审核后台

- 按待审核、无法判断、用户申诉查看队列。
- 同屏查看多视角图片、质量指标、模型输出和历史审核。
- 录入实际物种、真伪类别、炮制状态、鉴定依据和备注。
- 修改采用追加版本，不覆盖历史记录。

### 5.6 数据发布模块

- 专家确认数据先进入候选区。
- 发布前执行批次隔离、重复检查、授权检查和标签一致性检查。
- 每次发布生成不可变 manifest、split、数据指纹和审计报告。

## 6. 用户流程与状态机

用户流程：

```text
确认授权
-> 创建任务
-> 上传整根主图
-> 可选上传局部/断面
-> 服务端质量检查
-> 不合格则复拍
-> 闭集实验性分析
-> 查看结果或无法判断原因
-> 可申请专家复核
-> 查看专家结论
```

任务状态：

```text
draft -> ready -> queued -> analyzing
                           |-> completed
                           |-> needs_retake
                           |-> inconclusive
                           `-> failed
```

- `needs_retake`：输入属于目标流程，但图片质量不足。
- `inconclusive`：图片可处理，但策略拒绝给出真伪判断。
- `failed`：存储、队列、模型或程序异常；不得包含真伪结论。
- 专家审核是独立子状态，不改写原始模型预测。

## 7. 结果表达

阶段 A 允许三类用户结果：

- `正品特征吻合度高`：输入在闭集内更接近正品特征。
- `疑似伪品`：输入在闭集内更接近伪品特征，不能确认实际物种。
- `无法判断`：图片质量、输入分布或策略阈值不足以支持结论。

结果页必须显示：

- 本次实际参与推理的图片；
- 模型概率及其闭集条件解释；
- 未参与判断的局部或断面图片；
- 模型版本、策略版本和分析时间；
- 实验范围与专业鉴定提示；
- 复拍和专家复核入口。

专家结论与模型结论分栏显示，不能用专家结果覆盖历史模型输出。

## 8. 模型升级路线

阶段 C 模型由四项能力组成：

1. 图片质量：识别模糊、曝光、遮挡和主体完整度。
2. 主体定位：裁剪或掩膜药材区域，降低背景捷径。
3. 物种与真伪：区分正品、经验证的具体混淆品、其他冒充植物和未知样本。
4. 炮制质量：只在物种确认后判断合格、不合格或无法确认。

多视角模型使用共享视觉编码器处理 `whole`、`detail` 和 `cross_section`，以带缺失视角掩码的融合层生成任务输出。缺少视角不会触发程序错误，但会进入更保守的策略阈值。

开放集拒识使用独立 unknown calibration 与 unknown test。任何拒识阈值只能在 validation/unknown calibration 上选择，不能使用 locked test 调整。

## 9. 数据标签体系

核心字段：

| 字段 | 语义 |
|---|---|
| `sample_id` | 同一物理实物唯一编号 |
| `batch_id` | 最高隔离的来源或采购批次 |
| `capture_session_id` | 一次连续拍摄会话 |
| `view_type` | `whole`、`detail`、`cross_section` |
| `claimed_species` | 用户或包装声称物种 |
| `actual_species` | 专业确认的实际物种 |
| `species_verification` | `verified`、`unverified` |
| `authenticity_class` | `authentic`、`substitute`、`adulterant`、`unknown` |
| `processing_status` | `compliant`、`noncompliant`、`not_applicable`、`unknown` |
| `processing_issue_codes` | 由领域专家维护的具体不合格原因 |
| `verification_method` | 专家、显微、理化或分子鉴定 |
| `capture_device` | 设备型号或设备族 |
| `dataset_role` | train、validation、locked_test、ood_calibration、ood_test |

数据切分以 `batch_id` 为最高隔离单位。同一实物、批次或连续拍摄会话的全部视角只能进入一个分区。未验证的实际物种不能训练物种头，也不能由模型预测回填。

## 10. HTTP API

用户接口：

```text
POST   /api/v1/auth/register
POST   /api/v1/auth/login
POST   /api/v1/auth/logout
GET    /api/v1/auth/me
POST   /api/v1/cases
POST   /api/v1/cases/{case_id}/images
DELETE /api/v1/cases/{case_id}/images/{image_id}
POST   /api/v1/cases/{case_id}/analyze
GET    /api/v1/cases/{case_id}
POST   /api/v1/cases/{case_id}/review-requests
POST   /api/v1/cases/{case_id}/feedback
```

专家接口：

```text
GET  /api/v1/admin/reviews
GET  /api/v1/admin/reviews/{review_id}
POST /api/v1/admin/reviews/{review_id}/decisions
GET  /api/v1/admin/model-versions
```

`POST /analyze` 使用任务 ID 与输入图片指纹作为幂等键。重复提交返回同一分析任务，不重复排队。API schema 只允许向后兼容扩展；破坏性修改必须提升版本。

阶段 A 使用本地账户和可撤销的服务端不透明会话。密码只保存 Argon2id 哈希；会话 ID 通过 `HttpOnly`、`Secure`、`SameSite=Lax` Cookie 传输，状态变更接口同时校验 CSRF token。专家和管理员账户由管理员创建，不允许通过公开注册接口申请高权限角色。

## 11. 数据库与对象存储

主要数据表：

- `users`：用户、专家和管理员账户；
- `cases`：任务、所有者、状态和当前展示结果；
- `case_images`：对象键、视角、哈希和质量指标；
- `analysis_jobs`：任务键、重试、租约和错误；
- `predictions`：不可变模型原始输出和版本指纹；
- `review_requests`：用户复核申请；
- `expert_reviews`：版本化专业标签和依据；
- `model_versions`：模型文件、配置、数据和评估指纹；
- `consent_records`：分析授权与研究训练授权；
- `audit_logs`：安全和业务关键事件。

图片二进制只保存到 MinIO。对象键使用随机 UUID，不包含用户标识或原文件名。数据库保存对象键、哈希、尺寸、格式和授权状态。

## 12. 隐私与安全基线

- 上传压缩大小上限 20 MB，解码像素上限 4000 万。
- 允许 JPEG、PNG、WebP，但必须以实际解码结果为准。
- 保存前删除非必要 EXIF；原始文件名不进入对象键。
- 图片访问使用短期签名 URL，用户只能访问自己的任务。
- 用户、专家和管理员采用角色权限控制。
- 登录、上传、分析和审核接口分别限流。
- 浏览器认证使用安全 Cookie；生产环境启用 HTTPS、CSRF 和严格 CORS。
- 未授权用于研究的原图在任务完成 30 天后删除。
- 研究训练授权独立记录，用户撤销后阻止数据进入未来发布版本。
- 审核、模型晋升、数据发布和删除操作全部记录审计日志。
- 密钥只通过环境变量或秘密管理系统注入，不写入仓库和镜像。

## 13. 部署设计

阶段 A 使用单机 Docker Compose：

- Nginx：HTTPS 终止、静态资源和反向代理；
- Web：Vue 3/Vite/TypeScript 构建产物；
- API：FastAPI；
- Worker：Celery 与现有推理包；
- PostgreSQL；
- Redis；
- MinIO。

开发、测试和生产使用独立数据库、对象桶、队列命名空间和模型目录。模型只读挂载，Worker 启动时校验 SHA-256、schema、校准和范围标志。PostgreSQL 每日备份，MinIO 启用版本控制并定期执行恢复演练。

参考官方文档：

- Vue 3 + TypeScript：<https://vuejs.org/guide/typescript/overview>
- FastAPI 部署：<https://fastapi.tiangolo.com/deployment/>
- FastAPI 容器：<https://fastapi.tiangolo.com/deployment/docker/>
- PyTorch 迁移学习：<https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>

## 14. 测试策略

- 保留现有训练、切分、校准、推理、评估和晋升测试。
- 为任务状态机、幂等提交、权限判断、质量策略和结果格式编写单元测试。
- 为 PostgreSQL、Redis、MinIO、Celery 和推理 Worker 编写集成测试。
- 覆盖损坏图片、超限图片、伪造 MIME、解压炸弹、权限越界和过期签名 URL。
- 用固定黄金图片验证 Web 路径与 `InferenceService` 输出完全一致。
- 使用 Playwright 覆盖手机上传、复拍、任务恢复、结果查询和专家审核。
- 执行并发、Worker 崩溃恢复、重复消息和备份恢复测试。
- 阶段 C 按批次报告混淆矩阵、Macro F1、Balanced Accuracy、各类召回、AUROC、PR-AUC、Brier、ECE、NLL 和 bootstrap 95% 区间。
- 报告 unknown 错误接受率、拒绝率和拒绝后的选择性准确率。

## 15. 验收门槛

阶段 A：

- 原有闭集黄金图片结果无未解释变化；
- 系统错误绝不生成真伪结论；
- 上传、排队、重试、结果和审核可完整追溯；
- 未授权用户不能访问其他任务和图片；
- 在 4 vCPU、8 GB 内存的阶段 A 基线服务器上，模型预热后以 5 个并发任务测试，排除网络上传时间但包含排队、质量检查和推理，单图任务 P95 完成时间不超过 5 秒；
- 页面明确显示实验范围、模型版本和专业鉴定提示；
- 自动化测试、容器启动和恢复检查全部通过。

阶段 C：

- 模型指标必须在来源、批次和设备独立的手机 locked test 上报告；
- 伪品、具体混淆品、炮制不合格和 unknown 指标分别满足预注册门槛；
- 置信区间下界不满足门槛时不得晋升；
- test 不得用于调参、校准、选模或重复试验挑结果。

阶段 C 的具体数值门槛由首个真实手机 validation 数据集的基线和业务风险共同预注册，不在没有数据证据时虚构固定值。

## 16. 实施里程碑

| 阶段 | 预计时间 | 交付物 |
|---|---:|---|
| A1 | 1 周 | FastAPI、数据库、存储和队列基础 |
| A2 | 1 周 | 手机拍摄 PWA、上传和质量预检 |
| A3 | 1 周 | 推理 Worker、结果页和专家后台 |
| A4 | 1 周 | 安全、端到端测试、Docker 部署 |
| B | 6-12 周 | 手机多视角采集、专家审核和数据集版本 |
| C | 8-12 周 | 主体定位、多任务、开放集、多视角与晋升 |

## 17. 主要风险与控制

| 风险 | 控制 |
|---|---|
| 统一白底形成背景捷径 | 阶段 A 限定范围；阶段 B 引入真实背景；阶段 C 主体定位 |
| 二元伪品标签无法解释种类 | 不输出物种；只使用专家验证标签训练具体类别 |
| 同批次或同实物泄漏 | 以 `batch_id` 分区，多视角共享 `sample_id` |
| 类别不平衡与少数类区间过宽 | 扩充独立伪品批次，报告类别指标和置信区间 |
| 模型对未知药材强制分类 | 独立 OOD calibration/test 和拒绝策略 |
| 用户把概率理解为鉴定确定性 | 固定结果文案、范围标志和专家复核入口 |
| 在线数据污染训练集 | 候选区、专家审核、授权检查和不可变发布流程 |
| 多进程重复加载模型 | API 与 Worker 分离，每设备限制模型 Worker 数 |

## 18. 已确认设计决策

- 采用双轨渐进方案。
- 阶段 A 允许整根主图推理，局部和断面只留存与审核。
- 炮制质量与物种真伪拆为独立任务。
- API、推理 Worker、数据库和对象存储解耦。
- 用户结果、模型原始输出和专家结论分别保存。
- 阶段 A 使用单机 Docker Compose。
- 真实手机 locked test 是解除实验性标识的硬门槛。
