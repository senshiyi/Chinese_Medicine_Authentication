# 🕳️ Workspace 全局复利与踩坑日志

> **用途**：记录与 Claude Code 协作过程中的沟通分歧、理解偏差、修改回退和经验教训，实现"踩一坑、长一智"的复利效应。
>
> **维护方式**：当前协作 Agent 按下述维护规则记录；历史条目中的 Claude Code 名称和环境信息保留为事件证据。

---

## 有效偏好与历史复用规则（2026-09-08）

- 本轮明确指令优先；仅规划/审查/只读时不实施、不记日志。已有授权和已确定的方案不反复确认。
- 需要用户选择的是尚未确定的产品、平台、项目位置、组织方式、费用和数据去向等重要路线；内部等效实现按现有惯例决定。提问时同时给出选项、后果和推荐。
- 新项目父目录与组织方式未知时询问；已给路径或普通子目录直接沿用。输出位置与命名遵守适用项目规则。
- 保护原始资料、用户手改文档和未提交工作；覆盖无版本保护的文件前备份或另存。已有源码修复不因流程顺序重写或删除。
- 下方历史事件是带场景的证据，不是全局命令。旧日期的路径、版本、环境故障与临时绕行方法必须先核实现状；不要自动重放历史系统操作。
- 检索相关条目即可；同一任务已读且未变化内容可复用。重复事件更新一条，保留旧结论并标记修正关系。
- 2026-06-11 的“任何等效技术路径都确认”和 2026-06-17 的“任何目录必问”由本节收紧后的适用条件取代；保留用户对实际方案路线的决定权。

## 📋 日志条目模板

| 字段 | 说明 |
|------|------|
| **日期** | YYYY-MM-DD |
| **类别** | 沟通分歧 / 理解偏差 / 修改回退 / 环境问题 / 成功模式 / 其他 |
| **场景** | 当时在做什么任务 |
| **问题描述** | 发生了什么问题 |
| **根因** | 问题的根本原因 |
| **解决方案** | 最终如何解决 |
| **经验教训** | 以后如何避免 |
| **标签** | 便于检索的关键词 |

---

## 🗂️ 分类索引

### 🔀 沟通分歧
> Claude Code 理解有误或用户意图未表达清楚导致的来回沟通。

### 🤔 理解偏差
> Claude Code 对代码库、技术栈、业务逻辑的理解有偏差。

### ↩️ 修改回退
> 某次修改被证明是错误的，需要回退或重做。

### ⚙️ 环境问题
> 开发环境、依赖版本、路径等环境导致的问题。

### 🌟 成功模式
> 用户明确夸奖/认可某种处理方式，下次遇到同类场景应优先复用。

### 📌 其他
> 不属于以上分类但值得记录的问题。

---

## 📝 日志记录

> **排列规则**：按时间倒序排列，最新记录在最前面。同一天的记录优先按类别分组（沟通分歧 → 理解偏差 → 修改回退 → 环境问题 → 成功模式 → 其他）。

---

### 🎯 自动记录指令

实际纠正本次工作、明确认可可复用做法，或解决了值得复用的问题时，记录场景、证据与复用条件。关键词只作线索；引文、示例、否定及审计文本不触发纠错或记录。按 Workspace AGENTS 的任务边界执行，不在只读任务中写日志。

新条目插入下方 `<!-- new-experience-entry -->` 后，最新在前；同一事件避免重复。使用增量编辑，保留既有记录，禁止无备份全量重写。

---

### 🌟 成功模式记录模板

| 字段 | 说明 |
|------|------|
| **日期** | YYYY-MM-DD |
| **类别** | 成功模式 |
| **场景** | 当时在做什么任务 |
| **成功做法** | 具体做了什么让用户满意 |
| **为什么有效** | 区别于默认做法，好在哪里 |
| **复用条件** | 什么场景下可以复用（便于检索） |
| **标签** | 便于检索的关键词 |

### 2026-09-08 | 环境问题（DSH Desktop 便携打包：DSH_HOME、Defender 拦截、zip 条目分隔符）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-09-08 |
| **类别** | 环境问题 |
| **场景** | 把本机 DSH Desktop 2.0.3（程序在 `D:\WorkorStudy\DSHdesktop\DSH Desktop`，数据在 `C:\Users\29070\.dsh`）整理成新电脑（Windows 11）绿色便携压缩包。 |
| **问题描述** | (1) 程序目录与用户数据分离，直接拷程序不带已装插件/技能/预设；(2) 打包时 Windows Defender 按内容拦截 `@dhicoc/dsh-reverse-skill` 下若干攻击载荷参考 `.md`（读文件报“包含病毒/Invalid argument”，且同一文件时拦时不拦，robocopy 复制却能通过）；(3) PowerShell 5.1 的 `[System.IO.Compression.ZipFile]::CreateFromDirectory` 生成的 zip 条目名用**反斜杠**分隔，属非规范 zip，部分解压工具会把它当单个文件名导致解压错乱。 |
| **根因** | (1) DSH Desktop 通过 `home = homeDir ?? process.env.DSH_HOME ?? join(homedir(), '.dsh')` 决定数据根（已从 app.asar 确认），便携必须显式设 `DSH_HOME`；(2) Defender 启发式扫描安全技能载荷文本，robocopy 的打开方式能过、Get-FileHash/tar 的读取被拦；(3) .NET Framework 版 ZipFile 不做分隔符规范化。 |
| **解决方案** | (1) 包内放 `DSH Desktop/`（474MB）+ 精选 `.dsh/`（desktop profile 插件运行时 921MB、skills、.agent-presets、settings.yaml、skin-center 等，剔除 sessions/credentials/attachments/旧路径 pnpm 元数据 .modules.yaml），根目录放 `start_DSH_Desktop.bat`：`set DSH_HOME=%~dp0.dsh` 后再启动 exe；(2) 全量试读扫描 31689 个文件找出唯一被 Defender 持续拦截的文件并移除（注意要顺序扫描，PS5.1 无 `ForEach-Object -Parallel`、无三元运算符）；(3) 改用 `ZipArchive.CreateEntry` 手工写条目（条目名强制 `/` 正斜杠）+ 读取重试，0.9 分钟完成；解压校验 31688 文件数量一致、抽查 SHA256 全 MATCH。 |
| **经验教训** | Electron 应用做“程序+数据”便携包时，先查数据根的环境变量/默认路径逻辑再设计布局；打包含攻击载荷内容的技能库前先做一次 AV 拦截扫描；Windows 上大 zip 优先手动 `ZipArchive` 写正斜杠条目，别用 .NET Framework `ZipFile::CreateFromDirectory`（反斜杠坑）；本机 PowerShell 为 5.1（无 `-Parallel`/三元 `?:`）。 |
| **标签** | #DSHDesktop #便携包 #DSH_HOME #绿色安装 #Defender拦截 #zip反斜杠 #ZipArchive #Electron数据分离 #Windows11 |

---

<!-- new-experience-entry -->

### 2026-09-08 | 方案改进（Astra 指令与技能适用范围整理）

| 字段 | 记录 |
|------|------|
| **场景** | 用户先要求只读审计，随后明确授权按报告修改全部相关规则。 |
| **问题描述** | 无条件技能触发、重复设计审批、强制测试顺序和历史经验泛化导致潜在停工或扩展范围。 |
| **解决方案** | 保留项目事实、安全边界和用户方案选择权；收紧触发条件，复用已有授权，按风险验证，明确历史规则替代关系；修改前备份并保存差异。 |
| **复用条件** | 整理模型迁移后的持久指令时；先审计实际加载源，再按用户授权修改。插件缓存更新后按保存差异复核，不盲目覆盖新版。 |
| **标签** | #Astra #Skill范围 #授权复用 #比例验证 #历史经验 |

---

### 2026-09-03 | 环境问题（备份目录实际名称与用户输入路径不同）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-09-03 |
| **类别** | 环境问题 |
| **场景** | 根据用户提供的路径检查 ChatGPT/Codex 本机数据备份并验证会话是否可恢复。 |
| **问题描述** | 用户输入的 `E:\Desktop\chatgpt\_data\_backup\_2026\_09\_02\_204455` 按字面不存在；实际备份目录为 `E:\Desktop\chatgpt_data_backup_2026_09_02_204455`，目录名中的下划线不是层级分隔符。 |
| **根因** | 备份脚本将日期和数据标识拼接成单个目录名，用户消息中的反斜杠与下划线组合容易造成路径结构误读。 |
| **解决方案** | 先检查上级目录并按 `backup`、日期和时间特征匹配实际目录，再读取 `backup_manifest.json` 和数据库进行只读完整性校验。 |
| **经验教训** | 备份路径无法访问时，不要立即判断备份丢失；应先列出上级目录，核对实际目录名，再继续恢复检查。 |
| **标签** | #备份路径 #目录名核对 #ChatGPT #Codex #只读校验 |

---
### 2026-09-03 | 沟通分歧（恢复程序不能依赖同一桌面进程退出后继续）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-09-03 |
| **类别** | 沟通分歧 |
| **场景** | 恢复被运行中的 Codex 桌面端占用的本地会话文件。 |
| **问题描述** | 先要求用户退出 Codex 后再回复，以便继续恢复；但退出桌面端可能同时结束当前交互进程，用户指出“退出后怎么继续恢复”。 |
| **根因** | 没有把“释放文件锁”和“继续执行恢复”拆成两个独立生命周期，错误地依赖同一桌面交互继续存在。 |
| **解决方案** | 改为先部署独立的隐藏后台恢复程序；它等待 SQLite 文件锁释放，应用退出后自动复制完整备份并写入恢复日志，之后再由交互进程做校验。 |
| **经验教训** | 涉及当前桌面应用自身数据的恢复，必须先启动可脱离当前 UI 的后台任务，再要求用户关闭应用；不能让用户退出后再依赖原会话继续。 |
| **标签** | #沟通分歧 #后台恢复 #文件锁 #Codex #生命周期 |

---
### 2026-09-03 | 环境问题（长路径前缀变体未同步迁移）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-09-03 |
| **类别** | 环境问题 |
| **场景** | 用户打开恢复后的旧 Codex 会话时，提示带 \\?\ 前缀的 D 盘 rollout 文件不存在。 |
| **问题描述** | 普通 D 盘路径已迁移到 C 盘，但数据库中另有 19 条路径使用 \\?\D:\... 长路径前缀，未被普通前缀匹配规则覆盖。 |
| **根因** | Windows 同一文件路径存在普通路径和扩展长度路径两种字符串形式，路径迁移只处理了一种形式。 |
| **解决方案** | 确认对应 C 盘文件全部存在后，事务性将 19 条 \\?\D:\WorkorStudy\CodeX\CodexData\.codex 路径替换为 \\?\C:\Users\29070\.codex；随后成功读取目标会话。 |
| **经验教训** | Windows 路径迁移必须同时处理普通路径、\\?\ 扩展路径和可能的 UNC 变体，并逐条检查 rollout 文件存在性。 |
| **标签** | #Windows长路径 #扩展路径前缀 #rollout_path #SQLite #Codex会话恢复 |

---
### 2026-09-03 | 环境问题（会话数据库恢复后仍指向旧盘路径）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-09-03 |
| **类别** | 环境问题 |
| **场景** | 将 D 盘旧 Codex 数据备份恢复到当前默认的 C 盘 .codex 目录。 |
| **问题描述** | 数据库已恢复并通过完整性检查，但 threads.rollout_path 中的 248 条记录仍指向 D:\WorkorStudy\CodeX\CodexData\.codex\sessions\...，应用因此找不到旧会话内容。 |
| **根因** | 只复制了会话文件，没有同步修正 SQLite 中保存的绝对路径；数据库完整性正常不代表外部会话文件路径有效。 |
| **解决方案** | 在应用退出并释放锁后，重新复制完整会话文件，再批量将 threads.rollout_path 的旧根路径改为当前 C:\Users\29070\.codex，随后重新校验数据库和文件数量。 |
| **经验教训** | 跨目录恢复带有绝对路径的数据库时，必须同时检查并迁移数据库内的路径字段，不能只验证 SQLite integrity_check。 |
| **标签** | #路径迁移 #rollout_path #SQLite #Codex会话恢复 #D盘到C盘 |

---
### 2026-09-03 | 环境问题（恢复会话文件前必须释放 SQLite 文件锁）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-09-03 |
| **类别** | 环境问题 |
| **场景** | 将已校验的 Codex 会话备份恢复到当前默认数据目录。 |
| **问题描述** | Codex 进程仍占用 SQLite 主库、WAL 和共享内存文件；复制在覆盖 `state_5.sqlite-shm` 时被系统以“用户映射区域”错误拒绝，导致主文件已写入而旁车文件未写入。 |
| **根因** | SQLite 数据库由运行中的进程保持内存映射，不能在应用运行期间成套替换数据库文件。 |
| **解决方案** | 停止继续复制，保留安全副本；要求完全退出 Codex/ChatGPT 后，再一次性恢复主库、WAL、共享内存、会话目录和索引，并执行完整性校验。 |
| **经验教训** | 恢复 SQLite 会话数据前必须先关闭使用它的应用；不能只替换主库，也不能把主库与旧 WAL/SHM 混用。 |
| **标签** | #SQLite #WAL #SHM #文件锁 #Codex会话恢复 #应用退出 |

---
### 2026-09-02 | 环境问题（ChatGPT/Codex 数据备份清理脚本需避开 PowerShell 与 robocopy 参数坑）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-09-02 |
| **类别** | 环境问题 |
| **场景** | 用户要求将所有 ChatGPT/Codex 本机数据备份到桌面新文件夹，并清理备份外的本机数据。 |
| **问题描述** | 第一版备份脚本中，`[Parameter(Mandatory=$true)] [System.Collections.Generic.List[string]]` 在传入空集合时被 PowerShell 判定为不可绑定；修复后又因 `robocopy /LOG+` 的日志目录/日志参数处理不稳，导致所有复制返回 16。 |
| **根因** | PowerShell 参数绑定会把某些空集合视为无效 Mandatory 入参；`robocopy` 在日志路径父目录不可用或日志参数解析失败时会直接报 Invalid Parameter，并且脚本若吞掉标准输出会丢失关键诊断信息。 |
| **解决方案** | 去掉 List 参数的 Mandatory 标记；在复制前显式创建日志父目录；改用 `robocopy ... | Tee-Object -FilePath <log> -Append | Out-Null` 记录输出，而不是依赖 `/LOG+` 参数。备份成功后，再用独立后台清理脚本等待 ChatGPT/Codex 进程退出，避免运行中锁文件造成半清理。 |
| **经验教训** | 编写 Windows 备份/清理脚本时，先做只读盘点和备份 manifest；PowerShell 函数不要对可为空的集合入参加 Mandatory；`robocopy` 失败要保留完整输出日志；涉及当前运行中的 ChatGPT/Codex 数据目录时，清理阶段应等待进程退出后再删除。 |
| **标签** | #ChatGPT #Codex #本机备份 #数据清理 #PowerShell #robocopy #后台清理 #文件锁 |

---
### 2026-08-30 | 理解偏差（游戏重启后仍失败说明不是单纯继承代理状态）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-30 |
| **类别** | 理解偏差 |
| **场景** | 排查异环和三角洲在关闭 FlClash/梯子后无法正常联网，但浏览器可访问网页。 |
| **问题描述** | 初步认为可能是游戏/启动器在开梯子时启动，继承或缓存了 `127.0.0.1:7890` 代理状态；用户随后补充：关梯子后重新启动游戏仍然不行。 |
| **根因** | 过早把问题收敛到进程级缓存/继承代理，未充分区分“进程继承代理状态”和“游戏自身鉴权、网页、更新、反作弊或 CDN 链路直连不可达”。 |
| **解决方案** | 将假设调整为：基础浏览器链路正常，问题集中在游戏应用层网络；后续应在关梯子状态下抓取游戏进程连接、失败远端 IP/域名、启动器日志和具体报错，并优先判断哪些鉴权/网页/更新组件必须走代理或加速器。 |
| **经验教训** | 若关代理后浏览器正常但游戏重启仍失败，不要继续按“残留代理/旧进程缓存”单线排查；应转向游戏专属链路，尤其是 Electron/WebView/Booster 组件、Steam/WeGame 鉴权、反作弊服务和 CDN 连通性。 |
| **标签** | #FlClash #异环 #三角洲行动 #游戏网络 #启动器鉴权 #代理缓存误判 #应用层网络 |

---
### 2026-08-30 | 理解偏差（开着 FlClash 时不能证明关梯子后的直连状态）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-30 |
| **类别** | 理解偏差 |
| **场景** | 排查“只有开启 FlClash/梯子才能正常上网”时，用当前联网状态做直连测试。 |
| **问题描述** | 在用户仍开着 FlClash/梯子的状态下，依据 `curl --noproxy`、DNS 与百度连通测试判断“普通国内网络已经能直连”，用户随后指出当前仍处于开梯子状态。 |
| **根因** | 没有区分应用层代理绕过与系统层 TUN/DNS 接管；FlClash 开启时，即使命令绕过 HTTP 代理，底层流量或 DNS 仍可能被 TUN/SmartDNS 接管，因此不能证明关梯子后的真实网络状态。 |
| **解决方案** | 后续必须做 A/B 对照：分别记录“FlClash 开启”“仅关闭代理核心/TUN”“完全退出 FlClash”三种状态下的系统代理、WinHTTP、环境变量、DNS、默认路由、53 端口监听与国内站点连通性。若关梯子会导致聊天断连，应先生成本地诊断脚本，离线采集日志后再分析。 |
| **经验教训** | 网络排查里 `--noproxy` 只能绕过应用层 HTTP 代理，不能排除 TUN、DNS 劫持、WFP/虚拟网卡等系统层接管。用户说“现在开着梯子”时，应立即撤回对关梯子状态的判断，只保留当前状态证据。 |
| **标签** | #FlClash #TUN #DNS接管 #直连测试 #AB对照 #网络排查 #理解偏差 |

---
### 2026-08-30 | 环境问题（FlClash/SmartDNS 关闭后 DNS 指向本机导致断网）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-30 |
| **类别** | 环境问题 |
| **场景** | 用户反馈电脑只有开启 FlClash/梯子时才能正常上网，关闭软件或关闭梯子都会导致无法上网。 |
| **问题描述** | 系统代理和 WinHTTP 代理均为直连，但当前进程继承了 `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY=http://127.0.0.1:7890`；WLAN 的 DNS 为 `{192.168.110.36, 119.29.29.29}`，其中 `192.168.110.36` 正是本机 WLAN 地址。本机 53 端口由 `smartdns` 监听，FlClash 进程同时运行。 |
| **根因** | FlClash/SmartDNS 将系统 DNS 第一项指向本机地址做分流/防污染解析；FlClash 开启时本机 DNS 服务可用，关闭软件或关闭代理核心后，本机 DNS 服务失效，但网卡 DNS 仍优先查询本机，导致域名解析失败，表现为全局“无法上网”。 |
| **解决方案** | 先不要改路由；优先恢复 WLAN DNS：关闭 FlClash 的 DNS 接管/退出保留 DNS，或将 WLAN DNS 改回自动获取，或手动设为 `119.29.29.29`、`223.5.5.5`、`1.1.1.1` 等可用 DNS。若需要 ChatGPT/GitHub 等服务继续稳定访问，再单独保留 FlClash 系统代理或 TUN，而不是让普通上网依赖本机 SmartDNS 残留。 |
| **经验教训** | 排查“关梯子后全局断网”时按层验证：系统代理、WinHTTP、环境变量、DNS、默认路由、53 端口监听。若默认路由正常但 DNS 第一项是本机局域网 IP，且 53 端口由 `smartdns`/FlClash 相关进程监听，优先判定为 DNS 接管未恢复。 |
| **标签** | #FlClash #SmartDNS #DNS #关代理断网 #127.0.0.1:7890 #WLAN #Windows网络 |

---
### 2026-08-30 | 环境问题（ACE 重装后会恢复完整 AntiCheatExpert 组件）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-30 |
| **类别** | 环境问题 |
| **场景** | 用户此前清理过 ACE/WeGame/三角洲残留并重启确认干净，之后要求“再清理一遍 ace”。 |
| **问题描述** | 复查发现 `AntiCheatExpert` 又恢复为完整安装状态：存在 `ACE-CORE102706`、`ACE-CORE202706`、`ACE-CORE302706`、`ACE-SSC-DRV64`、`AntiCheatExpert Protection`、`AntiCheatExpert Service` 等服务/驱动注册项，同时存在 `C:\Program Files\AntiCheatExpert`、`C:\ProgramData\AntiCheatExpert` 和 `C:\Windows\System32\drivers\ACE-ADVT.sys`。 |
| **根因** | ACE 是会被相关游戏/启动器重新拉起的共享反作弊组件；只在某一时刻清掉单个驱动文件，不能保证后续不会被游戏重新安装。 |
| **解决方案** | 使用纯 ASCII 的 `cleanup_ace_only.ps1`，精确删除 ACE 服务/驱动注册项、`ACE-*` 系统驱动文件、`C:\Program Files\AntiCheatExpert`、`C:\ProgramData\AntiCheatExpert` 和 `HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\AntiCheatExpert`，并复查确认无服务、无驱动文件、无卸载项、无进程。 |
| **经验教训** | Windows 清理 ACE 时应避免宽泛匹配 `ACE`，否则会误伤 `Interface`、`Space` 等普通系统项；应排除 `EasyAntiCheat`，只匹配 `AntiCheatExpert`、`ACE-CORE`、`ACE-SSC`、`ACE-*` 驱动。若涉及 `.cmd/.ps1` 给用户手动管理员运行，脚本尽量纯 ASCII，中文路径用动态构造或反查方式处理。 |
| **标签** | #ACE #AntiCheatExpert #反作弊 #Windows驱动 #服务注册表 #纯ASCII脚本 #避免误匹配 |

---
### 2026-08-30 | 理解偏差（过早将 3503008 主因指向代理）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-30 |
| **类别** | 理解偏差 |
| **场景** | 排查 Steam 国服《三角洲行动》重装后仍报“游戏启动权限查询失败（3503008）”。 |
| **问题描述** | 看到 `df_launcher.exe` 通过 FlClash `127.0.0.1:7890` 建立大量连接后，过早把代理视为最可疑主因。用户后续证明：删除 WeGame 的三角洲数据后，在同一环境下曾成功越过 3503008，但随后出现启动器组件错误，退出后再次复发。 |
| **根因** | 将“请求经过代理”与“代理导致请求失败”等同，没有先完成相同网络环境下的状态对照。 |
| **解决方案** | 降低代理假设的优先级，转而对比清数据后的成功启动与随后失败启动。发现 Steam 启动器数据库被重建，同时全局 AntiCheatExpert 存在多个驱动服务注册与实际文件落点不一致、退出码 31 的证据，后续优先重置共享 ACE 组件。 |
| **经验教训** | 网络问题必须用直连/代理的对照结果确认，不能只看连接路径。对共用启动器和反作弊组件的游戏，“清数据后成功一次”应优先指向共享状态重建和组件注册冲突。 |
| **标签** | #三角洲行动 #Steam国服 #3503008 #WeGame共享数据 #AntiCheatExpert #代理误判 |

---

### 2026-08-29 | 理解偏差（误将 Steam 国区三角洲判定为 Steam 国际服）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-29 |
| **类别** | 理解偏差 |
| **场景** | 排查 Steam 启动《三角洲行动》时报“游戏启动权限查询失败（3503008）”。 |
| **问题描述** | 看到 Steam 日志中的网络国家从 US 切换到 CN，并结合 SteamDB 的国际包地区限制，过早把用户使用的版本归为 Steam 国际服；用户随后明确说明玩的是 Steam 国服。 |
| **根因** | 没有先核对官方开服公告和 Steam 国区的服务器分流规则，把 Steam 商店包限制等同于游戏实际服务器归属；同时误读了通用启动参数 `-gamechannel=3` 的含义。 |
| **解决方案** | 查阅 Steam 官方社区中的《全球 PC 不删档开服公告》，确认“Steam 国区进度与国服线上版本一致”；再查 Steam 配置确认 `-gamechannel=3` 是该 App 的默认通用启动参数，撤回“国际服锁区即根因”的判断，重新排查国服鉴权、网络和启动器缓存。 |
| **经验教训** | 遇到同一 Steam App 按账号或地区分流服务器的游戏，必须先核对发行方公告、账户区服和实际数据互通关系；Steam 包限制、客户端网络国家和游戏服务器归属不能互相替代。 |
| **标签** | #三角洲行动 #Steam国区 #国服分流 #3503008 #gamechannel3 #地区限制误判 |

---

### 2026-08-28 | 环境问题（安装 GitHub Skill 时 py launcher 指向失效 Python）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-28 |
| **类别** | 环境问题 |
| **场景** | 用户要求从 GitHub 安装 `codex-deepseek-subagent` Skill。 |
| **问题描述** | 按 Skill 文档用 `py -3` 运行安装脚本时失败，`py.exe` 仍指向不存在或不可用的 `C:\Users\29070\AppData\Local\Programs\Python\Python312\python.exe`；但 PATH 中的 `python` 来自 Anaconda，版本为 3.13.9，可正常运行安装脚本。 |
| **根因** | Windows Python launcher 的注册记录残留或损坏，与当前可用 Python 运行时不一致。 |
| **解决方案** | 不继续依赖 `py -3`，改用 `python <skill-installer>\scripts\install-skill-from-github.py --repo oil-oil/codex-deepseek-subagent --path codex-deepseek-subagent` 成功安装到 `D:\WorkorStudy\CodeX\CodexData\.codex\skills\codex-deepseek-subagent`。 |
| **经验教训** | 在 Windows 上安装 Codex GitHub Skill 时，若 `py -3` 报无法创建进程，先检查 `Get-Command python` 和 `python --version`；有可用 PATH Python 时直接用 `python` 运行 Skill Installer，避免被损坏的 `py` launcher 卡住。 |
| **标签** | #CodexSkill #GitHub安装 #pylauncher #Python #Anaconda #codex-deepseek-subagent |

---

### 2026-08-28 | 环境问题（PowerShell 7 正式安装与 UTF-8 profile 配置）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-28 |
| **类别** | 环境问题 |
| **场景** | 用户要求安装 PowerShell 7 并默认使用 UTF-8 编码。 |
| **问题描述** | `pwsh` 命令已能运行且显示 7.6.4，但实际来源是 Codex bundled runtime：`C:\Users\29070\.cache\codex-runtimes\...\pwsh.exe`，不是正式用户/系统安装；`winget` 不在 PATH；PowerShell 7 无 profile 时 `[Console]::InputEncoding` 仍是 `gb2312`。 |
| **根因** | 不能仅用 `pwsh --version` 判断 PowerShell 7 已正式安装，必须看 `Get-Command pwsh` 的 `Source`；Windows 控制台输入编码会继承本地代码页，PowerShell 7 的文件编码默认 UTF-8 并不等于交互输入也默认 UTF-8。 |
| **解决方案** | 使用 Microsoft 官方 `https://aka.ms/install-powershell.ps1` 脚本安装到用户目录 `%LOCALAPPDATA%\Microsoft\powershell`，用 `-AddToPath` 加入并前置用户 PATH；在 `D:\文档\PowerShell\profile.ps1` 写入 UTF-8 profile，显式设置 `[Console]::InputEncoding`、`[Console]::OutputEncoding`、`$OutputEncoding` 和常用输出 cmdlet 的默认 `-Encoding utf8`。 |
| **经验教训** | Windows 上安装/修复 PowerShell 7 时先查三件事：`Get-Command pwsh` 来源、`winget --version` 是否可用、无 profile 新进程的 Input/OutputEncoding。若 `pwsh` 来自工具自带 runtime，应装正式用户级版本并把 PATH 前置；若只改 OutputEncoding 而不改 InputEncoding，中文输入/管道仍可能受 GBK/GB2312 影响。 |
| **标签** | #PowerShell7 #pwsh #UTF8 #InputEncoding #OutputEncoding #PATH #winget缺失 #CodexRuntime |

---
### 2026-08-28 | 环境问题（ChatGPT/Codex 重装后稳定 bin 只补 codex.exe 导致 code-mode-host 缺失）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-28 |
| **类别** | 环境问题 |
| **场景** | 用户重装 ChatGPT/Codex 后打开任务仍失败，报 `%LOCALAPPDATA%\OpenAI\Codex\bin\codex-code-mode-host.exe` 不存在。 |
| **问题描述** | 之前修复时只把 `codex.exe` 放到稳定路径 `%LOCALAPPDATA%\OpenAI\Codex\bin\codex.exe` 并设置 `CODEX_CLI_PATH`，但 Codex 本地执行还会按 sibling 路径寻找 `codex-code-mode-host.exe`、`codex-command-runner.exe`、`codex-windows-sandbox-setup.exe`、`rg.exe`。这些文件只在 hash 子目录或安装包 `app\resources` 中存在，稳定 `bin` 根目录缺失。 |
| **根因** | 对 Codex 本地执行组件的依赖关系理解不完整：`CODEX_CLI_PATH` 指向稳定 `codex.exe` 后，相关 helper 也需要在同一稳定目录可被解析；只校验 CLI `--version` 不足以证明 code mode / command runner / sandbox helper 可用。另有第二层问题：用户级 `TEMP/TMP` 指向 `D:\Temp`，该目录所有者为 Administrators，当前用户只有 Modify，sandbox setup 无法追加 ACL。 |
| **解决方案** | 从当前 MSIX 包 `C:\Program Files\WindowsApps\OpenAI.Codex_26.825.4187.0_x64__2p2nqsd0c76g0\app\resources` 用流式复制补齐五件套到 `%LOCALAPPDATA%\OpenAI\Codex\bin`：`codex.exe`、`codex-code-mode-host.exe`、`codex-command-runner.exe`、`codex-windows-sandbox-setup.exe`、`rg.exe`，并逐一校验 SHA256。拒绝直接放宽整个 `D:\Temp` 权限，改为把用户级 `TEMP/TMP` 设置回 `C:\Users\29070\AppData\Local\Temp`。 |
| **经验教训** | 修复 ChatGPT/Codex 本地执行组件时不要只补 `codex.exe`，必须把 resources 中的完整执行组件同步到稳定 `bin` 根目录，并校验每个 helper 哈希。验证清单至少包含：目标路径存在、五件套哈希一致、`codex.exe --version`、`rg.exe --version`、关闭并重启 App 后普通 sandbox 命令可运行。遇到 sandbox ACL 失败时优先检查 `TEMP/TMP`，不要轻易给共享临时目录扩大 FullControl。 |
| **标签** | #ChatGPT #Codex #code-mode-host #command-runner #sandbox-setup #CODEX_CLI_PATH #bin组件缺失 #TEMP #ACL #重装修复 |

---
### 2026-08-28 | 环境问题（ChatGPT/Codex 本机全量备份遇到 EFS 属性与运行中 MSIX hive 锁）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-28 |
| **类别** | 环境问题 |
| **场景** | 用户要求把 GPT/ChatGPT/Codex 所有本机数据备份到桌面，稍后手动重装。 |
| **问题描述** | 主数据目录 `CODEX_HOME` 和 `%LOCALAPPDATA%\OpenAI` 成功备份，但兼容空目录 `%LOCALAPPDATA%\Codex`、`%APPDATA%\Codex` 因复制 EFS 加密属性报 `The specified file could not be encrypted`；MSIX 用户包目录中的 `SystemAppData\Helium\User.dat*` / `UserClasses.dat*` 因 App 正在运行被锁定。 |
| **根因** | `robocopy /COPY:DAT /DCOPY:DAT` 会复制目录/文件属性，遇到源目录 EFS 属性时目标盘无法加密会失败；Windows App 容器的 registry hive 文件在应用运行时被系统占用，普通复制无法读。 |
| **解决方案** | 第一轮保留完整主备份；第二轮 fallback 对兼容目录和 MSIX 包目录使用 `/COPY:DT /DCOPY:DT` 不复制属性，并对 MSIX 包目录排除 `User.dat`、`User.dat.LOG1/2`、`UserClasses.dat`、`UserClasses.dat.LOG1/2`。主 Codex 数据库仍以 `CODEX_HOME` 和 `%LOCALAPPDATA%\OpenAI` 为准，均已备份成功。 |
| **经验教训** | 备份 ChatGPT/Codex 本机数据时要区分“主数据目录失败”和“系统容器 hive/空兼容目录失败”。主数据优先保 `CODEX_HOME`、`%LOCALAPPDATA%\OpenAI`、环境变量/代理注册表快照；MSIX `SystemAppData\Helium` 锁文件可在关闭 App 后再补，或排除后记录说明。遇到 EFS 属性复制失败，改用不复制属性的 `/COPY:DT /DCOPY:DT`。 |
| **标签** | #ChatGPT #Codex #本机备份 #robocopy #EFS #MSIX #User.dat #文件锁 #桌面备份 |

---
### 2026-08-28 | 环境问题（Codex Windows sandbox helper 缺失导致普通 exec/apply_patch 被拦）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-28 |
| **类别** | 环境问题 |
| **场景** | 为“保留所有 Codex 数据并重新安装 ChatGPT 到 C 盘”生成保数据重装脚本时，需要读取工作区规则、检查安装包和创建脚本。 |
| **问题描述** | 普通 `exec_command` 一开始全部失败，报 `codex-windows-sandbox-setup.exe` 缺失；后续 `apply_patch` 也因同一 sandbox helper 缺失而无法读取/更新已创建的脚本文件。 |
| **根因** | 当前 Codex 桌面运行环境的 Windows sandbox setup helper 缺失或路径失效，导致依赖沙盒的命令和补丁工具无法启动。 |
| **解决方案** | 对只读检查和工作区内必要写入使用 `require_escalated` 的受控 PowerShell 命令；脚本修补时不用全量重写，采用 `Get-Content -Raw` + 精确 `Replace` + `Set-Content` 只替换单个字符串。 |
| **经验教训** | 遇到 `orchestrator_helper_launch_failed` 且提示 sandbox helper 缺失时，先判断是否为沙盒层问题；必要操作可走受控非沙盒命令，并保持最小范围。已有文件修改优先 `apply_patch`，若被同一环境问题阻断，则用 PowerShell 精确替换，避免全量重建。 |
| **标签** | #Codex #sandbox #apply_patch #exec_command #Windows #orchestrator_helper_launch_failed #精确替换 |

---
### 2026-08-28 | 环境问题（ChatGPT 桌面版打不开：chatgpt.com DNS 污染 + 出站未走代理）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-28 |
| **类别** | 环境问题 |
| **场景** | ChatGPT 桌面版（Codex Agent 版）打不开，后台有 ChatGPT.exe 进程但界面不出现、进程崩溃循环（多批不同 StartTime）。此前按"更新/版本不匹配"排查并未命中，真正根因在网络层。 |
| **问题描述** | 应用启动后内部 Codex app-server 无法就绪，`ChatGPT.exe` 反复退出重启。手动启动 `bin\codex.exe app-server` 时 stderr 报 `app-server remote control websocket loop stopped before client name was ready`（连 `https://chatgpt.com/backend-api/` 失败）。 |
| **根因** | **DNS 污染 + 出站未走代理**。实测：`chatgpt.com` 被解析到 `31.13.87.9`（Meta 的 IP，明显污染）；**直连 443 超时、ping 100% 丢包**；**走 Clash 代理（127.0.0.1:7890）能通**（返回 HTTP 403 = TCP+TLS 已建立，仅缺认证）。ChatGPT 的 app-server 需连 `chatgpt.com/backend-api` 认证/远端控制，直连被 DNS 污染挡住 → app-server 无法就绪 → 桌面打不开。 |
| **解决方案** | 让应用出站走 Clash：(1) 设**用户级**环境变量 `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY` = `http://127.0.0.1:7890`（Rust app-server 读这些）；(2) 系统代理 `HKCU\...\Internet Settings` 设 `ProxyServer=127.0.0.1:7890, ProxyEnable=1`（Chromium/Electron 层读这个）；(3) 结束崩溃循环进程后重启应用。若代理非全局/TUN 模式，必须设这些变量/系统代理，否则 app 直连必失败。 |
| **经验教训** | (1) Codex/ChatGPT 桌面版"打不开、后台有进程但进程崩溃循环"时，**先查网络**再看版本：`Resolve-DnsName chatgpt.com` 看是否解析到异常 IP（污染），`Test-NetConnection`/`ping` 看直连，`curl -x http://127.0.0.1:7890 https://chatgpt.com` 看是否代理能通。(2) 判据：**直连 000/超时 + 代理 403/200 = 必须走代理**；403 已说明网络通，只是缺认证，app 可正常继续。(3) 代理设置要**两层**都设：环境变量（`HTTPS_PROXY` 等，Rust 层读）+ 系统代理（`HKCU Internet Settings`，Chromium 层读）。(4) 环境变量只在**新进程**生效，设置后须完全退出并重启应用。(5) 与 07-22 那条"Codex 移动端远程控制 DNS 污染"同源：本机 DNS 污染 + 代理未接管是反复出现的共性问题。 |
| **标签** | #ChatGPT #Codex #打不开 #DNS污染 #chatgpt.com #代理 #Clash #HTTPS_PROXY #remote-control #backend-api #崩溃循环 #断网排查 |

---

### 2026-08-28 | 环境问题（ChatGPT 更新后 `CODEX_CLI_PATH` 版本不匹配再次打不开）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-28 |
| **类别** | 环境问题 |
| **场景** | ChatGPT 桌面版本次更新后又出现"打不开、后台有进程"的问题（与 07-28 同源，但暴露了 `CODEX_CLI_PATH` 需指向稳定路径的改进点）。 |
| **问题描述** | 应用从 `26.825.3734.0` 更新到 `26.825.4187.0` 后，`CODEX_CLI_PATH` 仍指向上一版解压的 `bin\<hash>\codex.exe`。应用 spawn 上一版 CLI → 版本/内容不匹配 → app-server 起不来 → 界面打不开但后台有 `ChatGPT.exe` 进程。 |
| **根因** | **版本内容更新，但 env 指向旧二进制**。更新后包内 `resources\codex.exe` 的 SHA256 已变化（`C93962A1…` → `34E9CFE7…`，版本号仍显示 `codex-cli 0.150.0-alpha.12.2`），而 env 指向的旧 CLI 未同步。应用靠 `CODEX_CLI_PATH` 定位可执行 CLI，指向旧版即失败。 |
| **解决方案** | (1) 对比新旧包 `resources\codex.exe` 的 SHA256，确认 env 指向的二进制已过时。(2) 把新版 `resources\codex.exe` **流式复制**到用户可执行目录（`[System.IO.File]::OpenRead` + `CopyTo`，绕开 `File.Copy` 的 EFS 加密报错），覆盖 `bin\<hash>\codex.exe` 并新增稳定路径 `bin\codex.exe`（无 hash）。(3) 更新 `CODEX_CLI_PATH` 指向 **`bin\codex.exe` 稳定路径**（写 `HKCU:\Environment`）。(4) 结束所有残留进程后重启验证（新版 CLI 实测可运行 `codex-cli 0.150.0-alpha.12.2`）。 |
| **经验教训** | (1) `CODEX_CLI_PATH` **应指向不带 hash 的稳定路径** `%LOCALAPPDATA%\OpenAI\Codex\bin\codex.exe`，而非指向带版本 hash 的子目录——这样 OpenAI 每次更新后二进制内容变了，路径不变，无需每次改 env。前提是尽快把新版 `resources\codex.exe` 复制过去刷新。(2) 判断 env 是否失效：用 `Get-FileHash` 对比 env 目标与当前包 `resources\codex.exe` 的 SHA256，不一致就是过期。(3) 复制大文件到 `bin\` 时 `File.Copy` 会报 `could not be encrypted`（EFS），改用 `OpenRead`+`CopyTo` 流式复制可绕过。(4) 更新后再打不开 = 优先查：`Get-AppxPackage OpenAI.Codex` 看新版本号 → 对比 SHA256 → 复制新版 CLI 到稳定 bin 路径 → 更新 env。 |
| **标签** | #ChatGPT #Codex #CODEX_CLI_PATH #MSIX #更新后打不开 #版本不匹配 #SHA256对比 #稳定路径 #bin_codex_exe #EFS #流式复制 |

---

### 2026-08-28 | 环境问题（git filter-repo 重写历史会覆盖未提交修改）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-28 |
| **类别** | 环境问题 |
| **场景** | B 档清理：git 历史瘦身（移除历史中的 Cache/ 等大文件） |
| **问题描述** | 用 `git filter-repo` 重写历史后，工作树被重置到新 HEAD，**未提交修改被覆盖丢失**：`项目/Data/experiments/drynaria_positive_only/metadata.csv`、`splits.json`、`PlanningWorldActionGameTests.java`、以及踩坑日志.md 中未提交的 2026-07-28 三条记录 |
| **根因** | filter-repo 结束时会对当前分支执行 checkout/reset 同步工作树；执行前只创建了 git bundle 备份（bundle 仅含已提交历史，**不含未提交修改**），未先 stash/commit 工作树 |
| **解决方案** | 已提交的丢失内容可恢复（踩坑日志 07-28 三条记录从会话记录原样找回）；未提交的 csv/json/代码修改无法恢复（从未进入 git，无快照）。正确流程：重写历史前**必须先 stash 或 commit 未提交修改**，再做 bundle 备份 |
| **经验教训** | 任何历史重写/回滚操作前，第一步是保证工作树干净（`git status` 无未提交修改）：要么 stash 要么 commit。bundle 备份 ≠ 未提交修改的备份。 |
| **标签** | #git #filter-repo #历史重写 #未提交修改丢失 #先stash #备份 |

---

### 2026-08-28 | 环境问题（GitHub LFS 免费配额超限导致 push 失败）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-28 |
| **类别** | 环境问题 |
| **场景** | B 档清理：git 历史瘦身后 force push 到 GitHub 远端 |
| **问题描述** | `git push --force` 报 `batch response: This repository exceeded its LFS budget`，LFS 对象（3 个 .pth，185MB）上传被拒，push 整体失败 |
| **根因** | 仓库用 Git LFS 管理 `*.pth` 模型权重（`.gitattributes` 配置），GitHub 免费账户 LFS 配额仅 1GB，历史累积已超限；重写历史后需要重新上传 LFS 对象时触发配额检查 |
| **解决方案** | 把 `*.pth` 从 git 历史移除（`git filter-repo --invert-paths --path-glob '*.pth'`），本地权重文件保留在 `Models/`；push 不再涉及 LFS 即成功。本地 `.git/lfs` 用 `git lfs prune` 清掉 185MB 孤儿对象 |
| **经验教训** | (1) GitHub 免费 LFS 只有 1GB，模型权重走 LFS 迟早撞配额，撞了以后所有 push 全挂；(2) 撞配额后的解药是**权重不入库**：留在本地 `Models/`（命名规范本就规定 Models 不提交），需要共享走网盘/GitHub Releases；(3) `git lfs prune --dry-run` 可先预览可清理对象。远端 GitHub 的 LFS 孤儿对象需在仓库 Settings → Storage 手动清理 |
| **标签** | #GitHub #LFS #配额超限 #push失败 #filter-repo #模型权重不入库 |

---

### 2026-07-28 | 环境问题（完整版·含纠正）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-28 |
| **类别** | 环境问题 |
| **场景** | ChatGPT 桌面版（含 Codex 编程 Agent 的 Windows AI 助手）更新后启动失败，报 `ChatGPT failed to start. Unable to locate the Codex CLI binary. Set CODEX_CLI_PATH or ensure the Electron resources include bin/codex.` |
| **问题描述** | 更新后 ChatGPT 打不开：先报"找不到 Codex CLI 二进制"，设置环境变量后又表现为"点击无反应但后台有进程"，重装前反复；最终通过"定位可执行 CLI + 干净重装到 C 盘（不清理数据）"解决。 |
| **根因** | 两层：(1) **打包 bug**：应用把 codex 二进制放到受保护目录 `C:\Program Files\WindowsApps\...\resources\`（`codex.exe` 310MB 等），但解析逻辑 `iV()`/`TV()` 期望 `resources\bin\codex` / `app.asar.unpacked\codex.exe`，两处都不存在，`$B()` 返回 null 抛错。(2) **执行权限**（更隐蔽）：即使指向 `resources\codex.exe` 也是死路——WindowsApps 目录由 TrustedInstaller 保护，普通进程 spawn 它时 Access denied（经对照测试：`cmd/notepad` 能跑，唯独 codex 不能）。应用又用 `cua_node` 运行时反复重试解压，残留多个 `.staging-*` 目录导致卡死。 |
| **解决方案** | (1) 应用会在用户可执行目录自行解压出 `%LOCALAPPDATA%\OpenAI\Codex\bin\<hash>\codex.exe`（与包内 `resources\codex.exe` SHA256 完全一致），把它设为**用户级** `CODEX_CLI_PATH`（`[Environment]::SetEnvironmentVariable(name,val,"User")`，写入 `HKCU:\Environment`），实测能运行（`codex-cli 0.150.0-alpha.12.2`）。(2) 卸载损坏的 MSIX 包 `Get-AppxPackage OpenAI.Codex | Remove-AppxPackage`，再用 `winget install --id 9PLM9XGG6VKS --source msstore` 干净重装到 C 盘 `C:\Program Files\WindowsApps`。(3) **全程不清理 `%LOCALAPPDATA%\OpenAI\Codex`**，登录态/会话数据保留。 |
| **经验教训** | (1) 报"找不到 CLI"时，先 `Get-AppxPackage "*OpenAI.Codex*"` 定位版本与安装位置，再查 `resources\` 下 codex 布局，判断是打包 bug。(2) **且必须验证选择的可执行文件真的能被运行**——用对照测试（跑 `cmd/notepad` 正常、唯独某 exe Access denied）区分"文件不存在"与"文件存在但无执行权限"；指向 TrustedInstaller 保护的 WindowsApps 路径是死路。(3) 环境变量方案要指向**用户可执行目录**里应用自行解压出的二进制（带 hash，与包内 SHA256 一致），且该路径含版本号，更新后需重新指向。(4) 治本是**干净重装**：`winget install --id 9PLM9XGG6VKS --source msstore`（msstore 源是桌面版，别误装 winget 源的 `OpenAI.Codex` CLI 命令行工具）。(5) **重装前先确认要不要清数据**：登录态在 `%LOCALAPPDATA%\OpenAI\Codex`，卸载包不删它，重装后无需重新登录。 |
| **标签** | #ChatGPT #Codex #CODEX_CLI_PATH #MSIX #Unable_to_locate_Codex_CLI_binary #更新bug #执行权限 #TrustedInstaller #干净重装 #winget #不清理数据 |

---

### 2026-07-28 | 成功模式

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-28 |
| **类别** | 成功模式 |
| **场景** | 商店 App（MSIX）更新后打不开，需判断"环境变量/路径补丁"还是"干净重装"，并做到不丢用户数据。 |
| **成功做法** | (1) 先并行查清三层事实：包版本与位置（`Get-AppxPackage`）、磁盘真实布局（`resources\` 有没有期望目录）、目标 exe 是否真能执行（对照测试跑 `cmd/notepad` 作基准）。(2) 用**可执行对照测试**戳破"文件存在=可用"的假象，识别出 TrustedInstaller 保护的死路路径。(3) 定位到应用自行解压到用户可执行目录的二进制（用 SHA256 与包内原件比对，确认同一版本）。(4) 用 `winget install --id <msstore ProductId> --source msstore` 干净重装到 C 盘，`--silent` 后台运行。(5) 重装全程**不清理用户数据目录**，卸载包不会删 `%LOCALAPPDATA%\OpenAI\Codex`，从而保住登录态/会话记录；装完验证包 Status=Ok + 环境变量目标可执行。 |
| **为什么有效** | 每条判断都有磁盘/进程/SHA256 证据，不靠猜；把"临时补丁（环境变量）"和"治本（重装）"分清楚，用户授权范围内做最小必要动作；优先保住用户数据，避免"修好了但登出/丢记录"的二次伤害。 |
| **复用条件** | 任何 MSIX/商店应用"更新后打不开/报找不到依赖组件"的场景；或涉及"受保护系统目录内文件不可执行"的排查。判断顺序：定位包 → 看布局 → 验证可执行 → 环境变量临时补丁 → 干净重装治本 → 全程保数据。 |
| **标签** | #MSIX #干净重装 #执行权限验证 #SHA256比对 #winget #不清理数据 #商店应用修复 #成功模式 |

---

### 2026-07-28 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-28 |
| **类别** | 环境问题 |
| **场景** | 用户 ChatGPT 桌面版（原 Codex 桌面 Agent 应用）启动报错 `ChatGPT failed to start. Unable to locate the Codex CLI binary. Set CODEX_CLI_PATH or ensure the Electron resources include bin/codex.` |
| **问题描述** | ChatGPT 桌面版启动即弹出失败对话框，提示定位不到 Codex CLI 二进制，建议设置 `CODEX_CLI_PATH` 或确保 Electron resources 包含 `bin/codex`。 |
| **根因** | **OpenAI 8-26 前后一次更新的打包 bug**：应用本体（CLI 组件）被安装为微软商店 MSIX 包 `OpenAI.Codex 26.820.7780.0`，二进制实际解包到了包的 `resources\` 根目录（`codex.exe` 307MB、`codex-code-mode-host.exe`、`codex-command-runner.exe`），但应用启动解析逻辑 `iV()`/`TV()` 期望的是 `resources\bin\codex`（或 `resources\app.asar.unpacked\codex.exe`）这一**旧版目录布局**，两处都不存在，`$B()` 解析返回 null 抛出该错误。`owl-app.ini` 中 `UserDataDirectoryName=Codex`，`%LOCALAPPDATA%\OpenAI\Codex\bin\` 目录存在但为空。 |
| **解决方案** | 设**用户级**环境变量 `CODEX_CLI_PATH`，指向应用自带的真实可执行文件 `C:\Program Files\WindowsApps\OpenAI.Codex_26.820.7780.0_x64__2p2nqsd0c76g0\app\resources\codex.exe`（PE 验证有效、MZ 头正确）。应用解析 `iV()` 被印证为：先读 `CODEX_CLI_PATH` 用 `existsSync` 校验存在即采用；含空格的完整路径不会被分词。设置后**完全退出并重启** ChatGPT 应用即可正常启动。 |
| **经验教训** | (1) ChatGPT 桌面版这样报错时，根因是更新把二进制放错目录布局，不是用户配置问题；先 `Get-AppxPackage "*OpenAI.Codex*"` 定位包、查 `package\app\resources\` 是否有 `codex.exe` 再判断。(2) 环境变量方案要设**用户级**（`[Environment]::SetEnvironmentVariable(name,val,"User")`，写入 `HKCU:\Environment`），指向 `resources\codex.exe`；该路径含版本号，OpenAI 后续更新后需同步更新。(3) 治本仍是干净重装一次或等官方修复。(4) 复制二进制到 `%LOCALAPPDATA%\OpenAI\Codex\bin` 会被两重障碍卡住：DSH 文件沙箱禁止写工作区外 + 源文件带 EFS 加密属性导致 `Copy-Item` 失败；兜底优先走环境变量路径。 |
| **标签** | #ChatGPT #Codex #CODEX_CLI_PATH #MSIX #Unable_to_locate_Codex_CLI_binary #更新bug #环境变量 #bin_codex缺失 |

---

### 2026-07-26 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-26 |
| **类别** | 沟通分歧 |
| **场景** | 中药真伪识别项目双任务架构与推理返回契约设计 |
| **问题描述** | 初版设计重点描述了药材类别、真伪类别、置信度和拒识原因等结构化字段，用户进一步明确最终展示内容应为“此药品为正品”一类自然语言结论。 |
| **根因** | 只从程序接口和未来 Web 复用角度设计返回值，没有同时显式定义终端用户看到的中文文案契约。 |
| **解决方案** | 将返回契约调整为“结构化 `PredictionResult` + 集中的 `ResultFormatter` 中文 `message`”；CLI 与未来 Web 默认展示自然语言结论，程序仍可读取标签、置信度和拒识原因。 |
| **经验教训** | 设计机器学习推理接口时，需要同时确认机器可读结构和用户可读结论；自然语言文案应集中生成，不能散落在 CLI、API 或页面代码中。 |
| **标签** | #机器学习推理 #返回契约 #中文结论 #Web预留 #ResultFormatter |

---

### 2026-07-22 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-22 |
| **类别** | 环境问题 |
| **场景** | 检查 `logs_2.sqlite` 是否因 TRACE 日志高频写入导致 WAL 持续增长 |
| **问题描述** | `D:\WorkorStudy\CodeX\CodexData\.codex\logs_2.sqlite` 在短采样中 `logs.max(id)` 快速增长，最近记录包含大量 `TRACE`（如 `codex_app_server::outgoing_message`、`codex_api::sse::responses`），WAL 文件持续有写入。 |
| **根因** | Codex 桌面端/相关服务仍在持续向 `logs` 表插入高频 TRACE 级别日志；仅 checkpoint 不能阻止后续 WAL 再增长。 |
| **解决方案** | 先用 SQLite Online Backup API 生成一致性备份，并额外保存 raw db/wal/shm；随后创建 `BEFORE INSERT ON logs` 触发器 `codex_block_logs_insert`，用 `SELECT RAISE(IGNORE)` 静默拦截日志插入；最后执行 `PRAGMA wal_checkpoint(TRUNCATE)` 并采样确认 `max(id)`、`sqlite_sequence` 和 WAL size 不再增长。 |
| **经验教训** | 遇到 Codex `logs_2.sqlite` / WAL 异常增长时，先采样确认 `logs.max(id)` 和最近日志级别；若确认为 TRACE 高频写入，备份后再加拦截 trigger，最后必须采样确认 MAX 与 WAL 稳定。 |
| **标签** | #Codex日志 #SQLite #TRACE高频写入 #WAL #trigger #checkpoint |

---
### 2026-07-22 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-22 |
| **类别** | 沟通分歧 |
| **场景** | Codex mobile 远程控制诊断结果交付 |
| **问题描述** | 用户此前长期使用中文交流，但本次诊断请求用英文书写，Codex 最终也用英文回答；用户随后追问“为什么没有用中文回答我”。 |
| **根因** | 语言匹配只机械跟随了最近一条英文任务，没有结合工作区历史教训、用户长期中文偏好，以及当前环境中“用户用中文时应中文回复”的明确经验。 |
| **解决方案** | 立即切回中文解释原因，并将本次场景记录为更细粒度的语言偏好规则。 |
| **经验教训** | 对该用户默认使用中文回复；即使用户偶尔用英文写技术任务，除非用户明确要求英文，最终诊断、交付和解释应优先中文。英文命令/错误日志可保留原文并用中文解释。 |
| **标签** | #语言匹配 #中文优先 #用户长期偏好 #英文任务中文交付 |

---
### 2026-07-22 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-22 |
| **类别** | 环境问题 |
| **场景** | Codex mobile 无法为本 PC 启用远程控制 |
| **问题描述** | 移动端提示无法启用远程控制/请重试。本机 Codex Desktop 与 remote_control enrollment 已存在且 enabled=1，但 app-server remote-control websocket 反复连接 `wss://chatgpt.com/backend-api/wham/remote/control/server` 失败。 |
| **根因** | Codex app-server 没有走本机 Clash 代理：WinHTTP 为 Direct，进程环境变量中也没有 HTTP(S)_PROXY/ALL_PROXY；同时路由器 DNS 将 `chatgpt.com` 解析到错误/不可达地址，导致直连 443 超时，日志出现 `os error 10060`。 |
| **解决方案** | 不改 Codex 数据库或关闭安全服务；最小修复是让 Codex app-server 的出站请求走可用代理（例如为启动 Codex 的环境设置 `HTTPS_PROXY/HTTP_PROXY/ALL_PROXY=http://127.0.0.1:7890` 或启用 Clash TUN/系统服务模式），或修复 DNS 后重启 Codex 并验证 remote-control websocket 不再超时。 |
| **经验教训** | Codex mobile 远程控制失败时，先查 `.codex\logs_*.sqlite` 中 `codex_app_server_transport::transport::remote_control` 日志，再做 DNS/代理对照测试；不要优先怀疑 Windows Remote Desktop/WinRM/RemoteRegistry 等系统远程服务。 |
| **标签** | #CodexMobile #remote_control #WebSocket #DNS污染 #代理未接管 #os_error_10060 #Clash |

---
### 2026-07-22 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-22 |
| **类别** | 沟通分歧 |
| **场景** | AI Player Company Mod 代码审查与 Bug 处理 |
| **问题描述** | 用户最初明确要求审查并修复现有 Bug，执行中途将任务范围改为只审查、不修改。此时 Mod 目录已有本轮未提交修复。 |
| **根因** | 用户在执行过程中调整了任务目标，属于需求范围变更，并非对初始指令的误读。 |
| **解决方案** | 立即停止修复，确认该 Mod 目录开始前无未提交差异后，仅将该目录恢复到 HEAD，验证目录差异为空；后续只提供审查结论与改进建议。 |
| **经验教训** | 用户中途缩小任务范围时，应立即停止写入并精确撤销本轮变更；日志需准确区分“用户变更需求”和“助手误解”，不能错误归因。 |
| **标签** | #任务范围变更 #只审查不修改 #精确撤销 #日志归因 |

---
### 2026-07-22 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-22 |
| **类别** | 环境问题 |
| **场景** | 按 `E:\Desktop\skill\skill.txt` 批量安装 Codex skills |
| **问题描述** | skill 清单来源混杂：部分是 openai/skills curated，部分来自 mattpocock/skills、obra/superpowers、kepano/obsidian-skills 等第三方仓库；GitHub Code Search API 未授权返回 401，部分 GitHub tree 请求返回 403；尝试 `npx skills add` 被安全策略拒绝。 |
| **根因** | `skill.txt` 只列 skill 名称和勾选标记，没有记录 repo/path；不同 registry 对同一 skill 有别名或旧路径，且执行第三方 npx 会引入未审计代码执行风险。 |
| **解决方案** | 优先使用已读过的官方 `skill-installer` 脚本，从可确认的 GitHub repo/path 安装；对括号别名建立映射（如 `diagnose`→`systematic-debugging`、`review`→`requesting-code-review`）；无法安全确认标准 `SKILL.md` 来源的条目列为未安装。 |
| **经验教训** | 批量安装 skill 时，清单最好包含 `name + repo + path + ref`；不要用第三方 npx 作为默认安装路径。遇到只有名称的清单，应先做来源对账，再安装可确认项，最后明确列出未覆盖项。 |
| **标签** | #skill安装 #CodexSkills #GitHub路径 #npx风险 #来源对账 |

---
### 2026-07-20 | 修改回退

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-20 |
| **类别** | 修改回退 |
| **场景** | Moisture determination drying-method video v2 voiceover revision |
| **问题描述** | User reported the v2 video still had only buzzing/background sound and no narration. The generated audio had fallen back to a synthetic test tone after TTS failed, but delivery validation only checked audio-stream duration and missed that it was not speech. |
| **根因** | The script allowed silent failure of TTS and used a tone fallback; verification relied on ffprobe metadata instead of confirming the audio source was a real voiceover file. |
| **解决方案** | Generated real Mandarin narration using Windows System.Speech with Microsoft Huihui Desktop, saved it as narration_voiceover_v3.wav, stretched the video to match the 102.6s narration, and muxed a v3 MP4 with only the real voiceover audio. Removed the tone fallback from the generator script. |
| **经验教训** | For voiceover deliverables, never include a buzz/tone fallback in formal output. If TTS fails, fail loudly. Validate the actual audio source and duration before muxing, not just that an audio stream exists. |
| **标签** | #voiceover #tts_failure #buzzing_audio #fail_loudly #video_revision |

---
### 2026-07-20 | 修改回退

| 字段 | 记录 |
|------|------|
| **日期** | 2026-07-20 |
| **类别** | 修改回退 |
| **场景** | Moisture determination drying-method teaching video |
| **问题描述** | The first video rendered Unicode subscript characters after m as square boxes, and the audio track was only simple background sound rather than real spoken narration. |
| **根因** | (1) Delivery validation checked metadata but did not sufficiently inspect rendered formula/font compatibility; (2) an audio stream was mistaken for voice explanation, without distinguishing background sound from narration. |
| **解决方案** | Generated a v2 video: replaced m subscript symbols with compatible m0/m1/m2 text; generated narration.wav with local speech synthesis; muxed it into the MP4 as the spoken explanation track; kept the old version instead of overwriting it. |
| **经验教训** | Before delivering teaching videos, verify that special formula characters render visibly, and when the user requests voice explanation, include actual understandable narration, not just background audio or a silent/non-speech track. |
| **标签** | #video #font_compatibility #subscript_boxes #voiceover #narration #revision |

---
### 2026-06-19 | 修改回退

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-19 |
| **类别** | 修改回退 |
| **场景** | 计算机网络实验报告：用户已将 10 张抓包截图插入了 .docx 报告，Claude Code 为补充数据表格而全量重新生成报告，直接覆盖了旧文件 |
| **问题描述** | 用户需要回退到旧版本，但旧 .docx（含截图）未提交 git、无任何备份，已被新版本覆盖。用户同时丢失了桌面上的 pcapng 抓包文件。所有截图需要重做。 |
| **根因** | (1) 生成新版本 .docx 时用相同文件名直接覆写，未先备份旧文件；(2) 未在覆盖前确认用户是否已保存/备份；(3) 项目未 git commit，无版本历史可回退 |
| **解决方案** | 无恢复路径。重做全部抓包并重新插入截图。 |
| **经验教训** | 覆盖任何用户可能已修改的文件前，**必须先备份**（如 `_backup_时间戳` 后缀），或生成到新文件名让用户确认后再替换。绝对不能假设"新版本更好所以覆盖没问题"——用户手工插入的内容（截图、修改）是无价的。 |
| **标签** | #覆盖文件 #未备份 #数据丢失 #.docx #先备份后覆盖 |

---

### 2026-06-19 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-19 |
| **类别** | 沟通分歧 |
| **场景** | 创建 Python_Programming 目录时，用户对两个问题的回答选"A,B"（Q1=A Q2=B，扁平结构），Claude Code 按此执行。随后用户自行判断后改为"是AA"。 |
| **问题描述** | 用户初始选择了 B（扁平结构），执行后自己判断应改为 A（标准嵌套结构），用"不对不对"纠正。Claude Code 事后记录的日志将责任归于自己解析有误，用户澄清是自己一开始判断错了，不是 Claude Code 的问题。 |
| **根因** | 用户在两次选择后自己调整了偏好，属于正常的决策迭代，不存在 Claude Code 误读。 |
| **解决方案** | 按用户修正后的 AA 重建结构。日志初稿错误归因已被修正。 |
| **经验教训** | 日志记录时要准确归因——用户自行改变主意 ≠ Claude Code 误读。不要为了"记一条教训"而把用户的责任揽到自己身上。 |
| **标签** | #日志归因 #用户自行纠正 #非Claude错误 |

---

### 2026-06-19 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-19 |
| **类别** | 沟通分歧 |
| **场景** | 用户要求为多元统计第十一次作业规划文件结构，Claude Code 规划完毕后主动提议"要开始写吗？" |
| **问题描述** | Claude Code 在文件结构规划完成后，顺带估计了篇幅并主动提议开始写作业内容。用户纠正：你只负责规划文件结构，不要越界到作业内容撰写。 |
| **根因** | 规划类任务完成后惯性延伸到执行阶段，未在任务边界处停下确认。用户的需求范围是明确的（"你帮我规划一下文件结构"），但 Claude Code 默认"规划完就问要不要做"，超出了用户的授权范围。 |
| **解决方案** | 立即确认边界，记录反馈。 |
| **经验教训** | 用户说"规划"就是规划，不要自动延伸到执行。任务完成 = 用户明确说的事做完，不等用户说"停"才停。做多余的事也是越界。 |
| **标签** | #任务边界 #规划≠执行 #不越界 #沟通 |

---

### 2026-06-16 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-16 |
| **类别** | 沟通分歧 |
| **场景** | cron 定时任务上午 11:00 触发学习提醒，Claude Code 未先 `date` 查钟点，直接按预设作息给出"继续睡"建议 |
| **问题描述** | 用户昨夜通宵（4:40 才睡），10:00 刚躺下。11:00 cron 触发时 Claude Code 直接假设正常作息，给了模糊建议。实际当时已是 14:28，用户已睡 10 小时。用户指出应先查时间再说话。 |
| **根因** | 收到 cron 预设提示后直接按常规逻辑回复，跳过了"先查时间"步骤。时间敏感判断依赖了预设假设而非实际数据。 |
| **解决方案** | 任何时间相关判断第一步先 `date`，第二步结合昨晚睡眠等上下文，再给建议。 |
| **经验教训** | 时间判断 = 先查钟 + 查上下文。cron 提醒只负责触发，不负责提供准确时间上下文。 |
| **标签** | #先查时间 #cron提醒 #默认假设 #时间判断 |

---

### 2026-06-17 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-17 |
| **类别** | 沟通分歧 |
| **场景** | 用户要学C语言，Claude Code 未询问位置偏好，直接在 `E:\Desktop\WorkSpace\C_Programming\` 创建项目目录和文件 |
| **问题描述** | Claude Code 按命名规范文档自动决定放 Workspace 根目录，但用户实际想把 C 语言放在 `Study_Helper\C\` 下，且目录结构也不是规范里的 Cache/Files/Models/References/Results，而是用户自定的 教材/作业/练习/题目 |
| **根因** | 项目放置位置和目录结构属于方案选择，不是纯技术细节。之前"先问后做"的经验（2026-06-11）未在"新建项目"场景下触发 |
| **解决方案** | 删除 `C_Programming/`，在 `Study_Helper/C/` 下按用户指定的四级分类重建 |
| **经验教训** | 新建项目/目录前必问两个问题：(1) 放哪个父目录？(2) 用什么目录结构？不能直接套命名规范——规范是默认值，用户可能有自己的组织体系 |
| **标签** | #新建项目 #先问后做 #目录位置 #Study_Helper #命名规范不覆盖用户偏好 |

---

### 2026-06-16 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-16 |
| **类别** | 沟通分歧 |
| **场景** | 排查凌晨4点自动睡眠任务，用户需提醒"把问题记到日志里"，Claude Code 才去读取踩坑日志并追加记录 |
| **问题描述** | AGENTS.md 第1-11行明确规定「每轮对话第一步必须先读取踩坑日志」，但 Claude Code 本轮对话启动后直接开始排查脚本/任务计划程序，未先读日志。用户发现问题解决后说"然后把之前的问题记到日志里"，Claude Code 才第一次打开日志文件。违反了 AGENTS.md 最高优先级指令。 |
| **根因** | 启动时看到明确的技术问题（"帮我看看有没有脚本"），直接进入排查模式，跳过了「先读日志+规范」的启动流程。AGENTS.md 指令是"每次对话开始时主动读取，不等用户提醒"，但实际行为是"有明确技术任务时优先执行任务，日志读了也用不上"。 |
| **解决方案** | 用户明确指出后立即纠正。核心教训：AGENTS.md 的"必读文件"不是「跟当前任务有关才读」，而是「不管什么任务都先读」。即使看起来无关，读了可能触发关联记忆。 |
| **经验教训** | 必读文件的优先级应该高于任何技术任务的执行优先级。不能因为"这个任务看起来跟日志无关"就跳过。正确的启动顺序：读 AGENTS.md → 读命名规范 → 读踩坑日志 → 再开始处理用户请求。 |
| **标签** | #启动流程 #必读文件 #AGENTS.md #优先级 #主动记录 #沟通 |

---

### 2026-06-16 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-16 |
| **类别** | 沟通分歧 |
| **场景** | 用户用中文发起对话，Claude Code 全程用英文回复 |
| **问题描述** | 用户说"Code"，Claude Code 用英文回复询问具体需求；用户继续用中文问"我可以和你聊一会天吗？为什么你是英文回答我"，Claude Code 才意识到应该用中文。用户需要主动提醒才切换到中文回复。 |
| **根因** | Claude Code 未根据用户的输入语言自动匹配回复语言，默认使用英文。即使对话中用户连续两轮都写中文，仍未能触发语言切换。 |
| **解决方案** | 用户明确指出后，Claude Code 切换到中文回复。 |
| **经验教训** | 用户用什么语言提问，就应该用什么语言回复。语言匹配是最基本的交互感知能力，不应等用户提醒"你为什么是英文"。每次对话开始时，第一轮就该根据用户输入的语言确定回复语言。 |
| **标签** | #语言匹配 #中文回复 #英文默认 #交互礼仪 #自动感知 |

---

### 2026-06-16 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-16 |
| **类别** | 环境问题 |
| **场景** | 排查电脑凌晨4点自动睡眠原因，发现 Claude Code 卸载后残留 Windows 定时任务 `AutoSleepAt4AM` |
| **问题描述** | 用户发现电脑每天凌晨4点自动睡眠，但找不到脚本位置。Claude Code 已卸载，`.codex` 目录已删除。用 `schtasks` 和 PowerShell `Get-ScheduledTask` 排查后，在 Windows 任务计划程序中找到 `AutoSleepAt4AM` 定时任务，触发时间每天 04:00，动作为 `rundll32.exe powrprof.dll,SetSuspendState 0,1,0`（强制睡眠）。 |
| **根因** | Claude Code 卸载时不会自动清理其创建的 Windows 计划任务程序条目，导致 `AutoSleepAt4AM` 残留在系统中，即使 Claude Code 已不存在任务仍继续执行。 |
| **解决方案** | `Unregister-ScheduledTask -TaskName "AutoSleepAt4AM" -Confirm:$false` 删除任务。同时发现 `PowerGridForecastTask`（路径 `\Microsoft\Windows\Sustainability\`，作者 Microsoft Corporation），确认为 Windows 系统自带的碳感知调度任务，非 Claude Code 残留，保留不删。 |
| **经验教训** | 卸载 Claude Code 后应检查 Windows 任务计划程序（`Get-ScheduledTask`）是否有带 `Claude Code`/`Auto` 关键词的残留任务。Claude Code 的自动化功能底层可能走系统 `schtasks`，卸载不会级联清理。 |
| **标签** | #Claude Code残留 #定时任务 #AutoSleep #schtasks #卸载清理 #SetSuspendState |

---

### 2026-06-11 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 沟通分歧 |
| **场景** | 创建定时任务时，Claude Code 未询问用户偏好，直接选择用 Windows 任务计划程序实现 |
| **问题描述** | 用户要求"设置一个定时任务"，Claude Code 发现 `automation_update` 工具不可用后直接退而用 `schtasks`，未告知用户有两种方案可选（系统任务计划程序 vs Claude Code 原生自动化），导致后续需要回退重做。 |
| **根因** | AGENTS.md 中"strongly prefer making reasonable assumptions"被过度执行——做技术选择时未区分"技术假设"（安全、无分歧）和"方案选择"（涉及用户体验、有替代方案需要用户取舍）。 |
| **解决方案** | 遇到二选一或多选场景（不同工具/方案/路径都可实现），先列出选项+后果，让用户选，不等指令直接做。 |
| **经验教训** | "合理假设"适用于纯技术细节（超时时间、编码格式），不适用于方案路线选择（用哪个系统、哪个工具）。后者必须问用户。 |
| **标签** | #方案选择 #沟通 #假设边界 #先问后做 |

---

### 2026-06-11 | 沟通分歧（复盘修正）

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 沟通分歧 |
| **场景** | 用户说"以后需要抉择的问我一下"，Claude Code 先记了日志、再单独列方案表，但用户真正意图是：每次遇到选择时，**同时**完成"问+列方案"两步，不等用户二次提醒。 |
| **问题描述** | Claude Code 把"问用户"和"给方案"拆成了两次独立响应，用户想要的是一个标准动作：抉择点出现 → 自动列出几个解决办法 → 让用户选。 |
| **根因** | 对用户指令做了过于字面的拆分：先记教训（"我会问"），再列方案（"这是选项"），而非一步合成"问+方案"。 |
| **正确做法** | 任何有分歧路径的节点，直接输出方案对比表+一句"你选哪个？"，不单独记教训、不单独许诺。 |
| **标签** | #先问后做 #方案对比 #合二为一 #理解偏差 |

---


### 2026-06-11 | 沟通分歧

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 沟通分歧 |
| **场景** | 用户要求"以后将文件修改前后有些什么地方改变了也要告诉我"，Claude Code 多次修改日志文件但从未主动告知变更内容 |
| **问题描述** | 本轮对话中 Claude Code 多次全量重写踩坑日志文件（排序、归类、新增条目），但每次只简要说明目的（如"fix ordering"），未列出文件具体改了什么。用户需要自己对比或追问才发现变更细节。 |
| **根因** | 默认行为是"做了就做了，只报告结果"，未形成"改文件 = 必出变更摘要"的习惯。 |
| **解决方案** | 每次修改文件后，用简短列表说明：哪些文件被改、每条改了什么（增/删/改/排序/归类）。 |
| **经验教训** | 用户不是盯着屏幕看你操作，变更摘要是最基本的协作礼仪。改了什么必须主动说，不等用户问。 |
| **标签** | #变更摘要 #文件修改 #沟通 #主动告知 |

---


### 2026-06-11 | 修改回退

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 修改回退 |
| **场景** | 用 PowerShell 字符串拼接方式精准插入新日志条目后，分隔符粘连 |
| **问题描述** | 插入的新条目末尾 --- 后缺少空行，与下一条 ### 粘连成 ---###，Markdown 渲染格式错乱。 |
| **根因** | 插入模板末尾只放了 --- + 换行，未补空行。Markdown 块间需要空行分隔。 |
| **解决方案** | 用 Replace 修复。后续插入模板末尾确保 --- 后跟两个换行。 |
| **经验教训** | 插入模板应自包含完整换行间距，不依赖插入点上下文。 |
| **标签** | #格式修复 #Markdown #换行缺失 #插入逻辑 |

---
### 2026-06-11 | 修改回退

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 修改回退 |
| **场景** | 多次用 `[System.IO.File]::WriteAllText` 全量重写踩坑日志文件，导致 2026-06-10 的日志条目被彻底覆盖丢失 |
| **问题描述** | 为修复日志排序问题，反复用全量重写方式重建文件。每次重建时只从当前文件内容提取条目，未意识到前一轮操作已经丢掉了 06-10 的日志。最终 06-10 的所有记录丢失，且无 git 历史、无文件备份可恢复。 |
| **根因** | (1) 编辑策略错误：修改文件内容应该用增量补丁（`apply_patch`），而不是全量重写；(2) 没有 git 版本管理兜底；(3) 每次重写前未核对条目总数是否一致。 |
| **解决方案** | (1) 初始化 git 仓库，将日志文件 + 命名规范 + .gitignore 纳入版本管理；(2) 日志文件新增「编辑方式要求」规则：优先用 `apply_patch` 增量修改，避免全量重写；(3) 06-10 条目需用户回忆内容后手动重建。 |
| **经验教训** | 编辑已有文件时，增量修改（apply_patch）优于全量重写。全量重写 = 每次都在赌自己记得所有内容，一旦忘记就永久丢失。git 是最后的安全网，但最优策略是不需要用到安全网。 |
| **标签** | #全量重写 #数据丢失 #apply_patch #git #增量优于全量 |

---

### 2026-06-11 | 修改回退

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 修改回退 |
| **场景** | 用户指出踩坑日志条目排列顺序不对，Claude Code 上次重建时声称"已按类别分组排序"但实际并未排序 |
| **问题描述** | 上次用 .NET 重建日志文件后，条目顺序仍是混乱的：环境问题 → 其他 → 踩坑解决 → 踩坑解决 → 沟通分歧 → 成功模式，完全不符合文件自身规定的「沟通分歧 → 理解偏差 → 修改回退 → 环境问题 → 成功模式 → 其他」分组顺序。 |
| **根因** | 上次重建时只关注了"把日志放到日志记录区域"，没有实际按类别排序，属于执行不完整。同时 `踩坑解决` 不是六类标准分类之一，归类也不规范。 |
| **解决方案** | 再次重建文件，严格按照类别分组排列（沟通分歧 ×2 → 修改回退 ×1 → 环境问题 ×2 → 成功模式 ×1 → 其他 ×1），并将两条 `踩坑解决` 重新归入标准分类（修改回退 / 环境问题）。 |
| **经验教训** | 执行"重建文件"类操作时，必须在完成后逐条验证排序和归类是否符合规则，不能只靠"感觉做好了"。排序规则是客观可检验的，写完就能自查。 |
| **标签** | #日志排序 #分类分组 #执行不完整 #自检 |

---

### 2026-06-11 | 修改回退

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 修改回退 |
| **场景** | 用 Windows 任务计划程序创建定时任务后，在 Claude Code 自动化界面中找不到 |
| **问题描述** | `schtasks` / `Register-ScheduledTask` 创建的是系统级 Windows 任务计划程序条目，Claude Code 有自己的自动化引擎，两者互不感知。 |
| **根因** | Claude Code 的 `automation_update` 工具未在当前会话加载，只能退而用系统工具；但系统工具创建的任务不会同步到 Claude Code UI。 |
| **解决方案** | 删除 Windows 任务，生成 Claude Code 自动化配置模板（`E:\Desktop\Claude Code自动化-桌面智能归档.md`），由用户在 Claude Code 界面中手动创建。脚本本身（`smart-archive.ps1`）无需改动。 |
| **经验教训** | 涉及 Claude Code 原生功能（自动化/线程/插件）时，先确认对应工具是否可用；不可用时生成配置模板让用户手动操作，不要用系统工具替代。 |
| **标签** | #Claude Code自动化 #schtasks #工具不可用 #模板方案 |

---




### 2026-06-11 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 环境问题 |
| **场景** | Claude Code 无法新建对话，点击无反应，恢复对话失败，无法归档 |
| **问题描述** | 关闭 Claude Code 界面后仍有 12 个后台进程残留，state_5.sqlite 被锁死，4 个 WAL 文件无法合并。旧对话正常（内存中已加载），但任何需要读写数据库的操作（新建/恢复/归档）全部失败。 |
| **根因** | Claude Code 某次异常崩溃后进程清理不完整——主进程退出但子进程（codex-plus-plus 等）继续运行并持有数据库写锁。后续重启无法获取锁。 |
| **解决方案** | 重启电脑释放所有进程和文件锁。重启后恢复正常。 |
| **经验教训** | Claude Code 出现"无法新建对话但旧对话正常"时，先检查：(1) 后台进程数是否异常多；(2) sqlite 文件是否被锁。若锁死，重启电脑是最快解法。 |
| **标签** | #Claude Code崩溃 #sqlite锁定 #进程残留 #WAL #重启解决 |

---
### 2026-06-11 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 环境问题 |
| **场景** | 用 PowerShell here-string 内联脚本做精准插入时，脚本中出现引号冲突导致解析失败 |
| **问题描述** | PowerShell 命令行中 '...' 和 "..." 与内容中的引号嵌套冲突，报 Unexpected token 和 string is missing the terminator。内联脚本中同时包含 "、'、Markdown 反引号时极易触发。 |
| **根因** | 内联 PowerShell 的引号转义规则复杂——内容中已含多层引号时，靠转义难以保证正确性。 |
| **解决方案** | 将脚本写入临时文件 $env:TEMP\insert_log.ps1，再用 powershell -File 执行。文件中的 here-string 不受命令行引号解析影响，彻底避开转义问题。 |
| **经验教训** | 复杂 PowerShell 脚本（含多层引号的 here-string）不要内联在 shell_command 中，写 .ps1 文件执行更可靠。该模式可复用。 |
| **标签** | #PowerShell #引号冲突 #here-string #脚本文件 #转义 |

---
### 2026-06-11 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 环境问题 |
| **场景** | 尝试用 pply_patch 增量修改日志文件，工具不可用 |
| **问题描述** | 调用 pply_patch 返回 unsupported call，当前会话工具集不包含该工具，只能退而用 PowerShell 字符串操作。 |
| **根因** | 当前 Claude Code 会话实际可用工具以调用返回为准，prompt 中提及不代表一定会加载。 |
| **解决方案** | 用 PowerShell IndexOf + Substring 实现精准插入，避免全量重写。 |
| **经验教训** | 不能假设 prompt 中提到的工具一定可用，需确认后再用。不可用时 PowerShell 字符串操作是安全的替代方案。 |
| **标签** | #apply_patch #工具不可用 #PowerShell替代 |

---
### 2026-06-11 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 环境问题 |
| **场景** | `schtasks /Create` 成功后出现密码警告 |
| **问题描述** | 警告 `When the run-as password is empty, the scheduled task may not run because of the security policy`，表示任务可能因安全策略无法执行。 |
| **根因** | `schtasks` 创建任务时未提供 `/RP`（密码），某些 Windows 安全策略禁止空密码用户执行计划任务。 |
| **解决方案** | 删除 `schtasks` 创建的任务，改用 PowerShell cmdlet：`New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive`（交互式登录，无需存储密码），配合 `Register-ScheduledTask` 注册。不设 `-RunLevel Highest` 以避免权限问题。 |
| **经验教训** | 用户级定时任务最佳实践：PowerShell `New-ScheduledTaskPrincipal` + `-LogonType Interactive`，无需密码也不触发安全策略警告。 |
| **标签** | #定时任务 #密码策略 #LogonType #schtasks |

---

### 2026-06-11 | 环境问题

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 环境问题 |
| **场景** | 用 Python 脚本重写踩坑日志文件，文件路径包含中文字符 |
| **问题描述** | `open(r"E:\Desktop\WorkSpace\Workspace全局复利与踩坑日志.md", 'w')` 报 `OSError: [Errno 22] Invalid argument`，Python 无法打开含中文路径的文件。 |
| **根因** | Windows 下 Python 的 `open()` 对含非 ASCII 字符的路径处理不稳定，尤其是通过 stdin 传入的 here-string 脚本中。 |
| **解决方案** | 改用 PowerShell 的 `[System.IO.File]::WriteAllText($filepath, $content, [System.Text.UTF8Encoding]::new($true))`，路径作为 PowerShell 变量传递，不受 Python 编码限制。 |
| **经验教训** | 在 Windows 上处理含中文路径的文件写入时，优先用 .NET 的 `System.IO.File` 方法或 PowerShell 原生命令，避免 Python `open()`。如果必须用 Python，应通过 `sys.argv` 传路径而非内联在脚本中。 |
| **标签** | #Python #中文路径 #OSError #System.IO.File #编码 |

---

### 2026-06-11 | 成功模式

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 成功模式 |
| **场景** | 用户要求"遇到选择时先问"，Claude Code 接下来列出三选一方案表让用户拍板 |
| **处理方式** | 用表格列出 Claude Code 原生自动化、Windows 任务计划程序、两者并存三个方案，每个附优缺点，等待用户选择后再执行。 |
| **效果** | 用户认可交互方式。 |
| **复用条件** | 任何技术实现有两种及以上等效路径时，用表格列出选项+优缺点+推荐，等用户确认后再动手。 |
| **标签** | #方案选择 #先问后做 #表格对比 |

---

### 2026-06-11 | 其他

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-11 |
| **类别** | 其他 |
| **场景** | 用户发现踩坑日志排列混乱，前两条 2026-06-11 的日志跑到了文件最顶部（标题之前），日志记录区域内也没有严格时间倒序 |
| **问题描述** | 之前追加日志时直接插入到文件最前面，而非定位到「日志记录」区域内部。导致标题和模板区域被日志条目夹在中间，且同一天内的日志也未按时间+类别排序。 |
| **根因** | 追加日志时没有先定位「日志记录」标题的位置，直接用了"插到文件最前面"的粗暴逻辑。 |
| **解决方案** | 用 [System.IO.File] 重建整个文件：(1) 标题+模板+分类索引+指令在前；(2) 所有日志条目统一放在「日志记录」区域；(3) 按时间倒序排列，同天按类别分组（沟通分歧 → 理解偏差 → 修改回退 → 环境问题 → 成功模式 → 其他）。 |
| **经验教训** | 追加日志前必须先定位插入点（「日志记录」标题），不能简单 prepend。该逻辑应作为"踩坑日志追加"操作的核心步骤固化。 |
| **标签** | #日志追加 #排序规则 #分类分组 #插入位置 |

---

---

### 2026-06-14 | 踩坑 | CODEX_HOME 环境变量残留导致桌面端无法启动

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-14 |
| **类别** | 踩坑解决 |
| **场景** | 安装 Claude Code 桌面端后启动报错：CODEX_HOME points to D:\WorkorStudy\...，path does not exist |
| **问题描述** | 从 C→D→E 盘迁移后，Machine 级 CODEX_HOME 在内存中残留 D 盘旧值（注册表已删除），覆盖了 User 级 E 盘正确值。 |
| **根因** | Windows Machine 级环境变量改后需重启才清除内存缓存，Machine > User 优先级。 |
| **解决方案** | setx 设 User 级为 E:\Desktop\WorkSpace\.codex，重启清除 Machine 缓存。Junction ~/.codex→E盘桌面端不认。 |
| **教训** | 环境变量迁移后必须重启；Machine 级谨慎使用，优先 User 级。 |
| **标签** | #CODEX_HOME #环境变量 #Machine级 #内存缓存 #桌面端 #重启 |

### 2026-06-14 | 踩坑 | Claude Code++ 桌面端不跟随 NTFS Junction

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-14 |
| **类别** | 踩坑解决 |
| **场景** | 尝试用 mklink /J 重定向 ~/.codex 到 E 盘 |
| **问题描述** | Junction 创建成功但 Claude Code++ 桌面端不跟随，mklink /D 符号链接需管理员权限。 |
| **根因** | 桌面端对路径做规范化处理，不认 Junction。 |
| **解决方案** | 放弃 Junction，依赖 CODEX_HOME 环境变量定位数据目录。 |
| **标签** | #Junction #桌面端 #Claude Code++ #重定向失败 |

### 2026-06-15 | 踩坑 | Obsidian 知识网络搭建：图片路径与 MOC 命名一致性

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-15 |
| **类别** | 理解偏差 |
| **场景** | 在 Obsidian vault 中批量创建 18 门课程的知识网络 MOC |
| **问题描述** | 两个问题：(1) 数学分析用 `工科数学分析学习流程.md` 作为实际 MOC，但文件名不符合 `XXX MOC.md` 规范，与其他科目不一致；(2) 图片文件夹 `picture/` 放在 vault 根目录，应该按科目归类到对应文件夹下 |
| **根因** | 数学分析是最早创建的笔记，当时没有统一的 MOC 命名规范；图片在搭建知识网络前就已存在，迁移时没考虑路径重构 |
| **解决方案** | (1) 新建 `数学分析 MOC.md` 作为统一入口，原 `工科数学分析学习流程.md` 保留作为详细路线图，两者互相链接；(2) 将 `picture/` 移入 `数学分析/picture/`，每个科目各自管理自己的图片 |
| **经验教训** | 批量创建知识网络前，先定命名规范（所有科目 MOC 统一 `XXX MOC.md`）和资源目录规范（`科目/picture/`），再开始铺文件。不要边建边改 |
| **标签** | #Obsidian #知识网络 #MOC命名 #图片路径 #目录规范 |

### 2026-06-16 | 踩坑 | Windows .bat 文件 UTF-8 中文乱码导致命令行解析错误

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-16 |
| **类别** | 踩坑解决 |
| **场景** | 桌面快捷方式启动 `Claude_学习助手.bat` 和 `Claude_通用对话.bat` 时报错，中文 echo 变成乱码，部分字符串被当成命令执行 |
| **问题描述** | 学习助手报 `'Math_Analysis_notes' 不是内部或外部命令`、`'╂墜' 不是内部或外部命令`；通用对话报 `'1' 不是内部或外部命令`、`'ude' 不是内部或外部命令`。根本原因是 .bat 文件以 UTF-8 编码保存（含中文字符），但 Windows cmd.exe 默认用 GBK（CP936）解析，UTF-8 多字节序列被拆解为乱码，部分字节恰好对应 `\|&<>` 等 cmd 特殊字符，导致命令行结构被破坏 |
| **根因** | (1) `.bat` 文件用 UTF-8 without BOM 保存，cmd.exe 不认；(2) 即便 `chcp 65001` 切换了控制台编码，.bat 文件本身的解析发生在 `chcp` 执行之前，此时文件已在 GBK 下被错误解析 |
| **解决方案** | 两个脚本全部重写为纯 ASCII（无中文字符）：学习助手用英文 echo 替代中文提示；通用对话改为一行 `start powershell -NoExit -Command "cd E:\Desktop\WorkSpace; claude"` 直接启动 PowerShell |
| **经验教训** | Windows `.bat` 文件不要写中文——要么用纯 ASCII，要么改用 `.ps1`（PowerShell 原生支持 UTF-8）。`chcp 65001` 改不了文件被解析时的编码，它只影响后续命令的输出 |
| **标签** | #bat乱码 #UTF-8 #GBK #cmd #PowerShell #桌面快捷方式 |

---

### 2026-06-16 | 踩坑 | 作息表与数学分析计划中日期-星期系统性偏移

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-16 |
| **类别** | 理解偏差 |
| **场景** | 用户说"今天是周二，6月16日"，发现作息表中 6/16 标的是周一。排查后确认过渡周（6/16–6/21）所有星期全部偏移一天 |
| **问题描述** | 作息表 6/16 标周一（实为周二），数学分析每日学习计划第 1 周 6 天全部标错（周一→周二偏移）。第 2 周起反而是对的——因为原计划把 6/16 当周一，6/22 碰巧也是周一所以对齐。共影响 2 个文件、19 处日期-星期组合（作息表 13 处 + 数学分析 6 处） |
| **根因** | 创建日程时未查日历确认 6/16 是周几，直接假设 6/16 是周一。后续所有日期-星期按此错误基准推算，导致过渡周全体偏移 |
| **解决方案** | 用 2026 年 6 月真实日历（6/1=周一 → 6/16=周二）统一修正所有文件：作息表过渡周+考试周全部修正，数学分析第 1 周全部修正 |
| **经验教训** | 创建含日期-星期的计划时，第一步必须查日历确认基准日是周几。一个基准日错误会连锁污染后续所有日期。不能靠"感觉"或假设 |
| **标签** | #日期星期匹配 #日历基准 #偏移连锁 #计划创建 |

---

### 2026-06-16 | 踩坑 | Obsidian MathJax 渲染 `\begin{cases}` 分段函数失败

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-16 |
| **类别** | 环境问题 |
| **场景** | 给用户出数学分析连续性题目，用 `\begin{cases}` 写分段函数，Obsidian 渲染出来两个分支挤在同一行 |
| **问题描述** | LaTeX 标准写法 `\begin{cases} ... \\ ... \end{cases}` 在 Obsidian 的 MathJax 中不生效，`\\` 换行被忽略，两行合并为一行 |
| **根因** | Obsidian 内置 MathJax 对 `cases` 环境的换行支持不完整，具体版本或配置差异导致 `\\` 不换行 |
| **解决方案** | 改用 `\begin{array}{ll} ... \\ ... \end{array}` 配合 `\left\{` 和 `\right.` 手动构造分段函数大括号，渲染正常 |
| **经验教训** | 在 Obsidian 中写分段函数，优先用 `array` + `\left\{` 而非 `cases`。出题前先确认目标渲染环境 |
| **标签** | #Obsidian #MathJax #cases渲染 #分段函数 #LaTeX兼容 |

---

### 2026-06-16 | 成功模式 | 英语每日计划从 pending 落地为可执行方案

| 字段 | 记录 |
|------|------|
| **日期** | 2026-06-16 |
| **类别** | 成功模式 |
| **场景** | 用户问"英语应该做点什么"，英语 MOC 处于 pending 状态，只有宏观大纲没有每日执行方案 |
| **成功做法** | (1) 先确认用户已有工具（奶酪单词），不强行推荐新工具；(2) 针对 CET-4 阶段明确各板块优先级（词汇>阅读>听力>写作>口语），写作和口语延后；(3) 30min 拆分为 15min 词汇雷打不动 + 15min 阅读/听力隔天轮换；(4) 纸笔/无纸化不做二选一，各取所长（背词无纸化、抄例句纸笔）；(5) 更新 MOC status 从 pending → active，每日计划写入文件固化 |
| **为什么有效** | 先问用户有什么再设计，不另起炉灶；30min 轻量可每天坚持；承认"CET-4 阶段不搞写作口语"降低了心理负担；用户说"可以"即确认，不反复打磨 |
| **复用条件** | 将 pending 状态的计划落地时——先确认用户已有资源/工具，再按优先级裁减到可每日执行的最小单元，最后写回 MOC 固化 |
| **标签** | #英语学习 #计划落地 #pending→active #MOC更新 #轻量坚持 |

### 2026-08-30 | 踩坑 | 游戏必须开梯子才能玩（浏览器正常）——残留代理环境变量

| 字段 | 记录 |
|------|------|
| **日期** | 2026-08-30 |
| **类别** | 环境问题 |
| **场景** | 用户反映国服《异环》《三角洲行动》必须开着梯子（FlClash）才能玩，关掉就报错（异环 -306 网络错误 / 三角洲 3503008 启动权限查询失败），但浏览器直连一切正常 |
| **问题描述** | 浏览器正常、仅游戏连不上网，且梯子一开就好——典型的"游戏和浏览器走了两套网络配置"现象 |
| **根因** | 用户级环境变量残留 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY = http://127.0.0.1:7890（FlClash 代理端口）。游戏/启动器/反作弊读环境变量→所有 HTTP 请求发给 127.0.0.1:7890；梯子关时端口无监听→连接失败。浏览器不读环境变量、只认系统代理（当时是关的）→直连正常 |
| **解决方案** | [Environment]::SetEnvironmentVariable('HTTP_PROXY',,'User') 删除三个变量（需写 HKCU\Environment，沙箱拦截时升级 danger-full-access 并获用户同意），删除后重启游戏/电脑生效 |
| **经验教训** | 排查"浏览器正常但应用连不上网"：第一优先查用户级环境变量 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY 是否残留指向代理端口；netstat 看到 SYN_SENT 卡在 127.0.0.1:7890 的进程（如游戏进程）即现场实锤 |
| **标签** | #代理环境变量 #HTTP_PROXY #游戏连不上 #FlClash #浏览器正常应用异常 #netstat排错 |

