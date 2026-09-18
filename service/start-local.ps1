# =============================================================================
# start-local.ps1 — Windows 本地预览一键启动（无 Docker）
#
# 启动三个服务（后台隐藏进程，日志写到 Cache/local-services）：
#   * API     127.0.0.1:8010  （venv-stage-a 解释器，uvicorn）
#   * Web     127.0.0.1:5174  （Vite 开发服务器，/api 代理到 8010）
#   * Worker  本地轮询 Worker （只读使用原项目 .venv/python.exe 的 Conda Python，
#             -B 防止在原项目写字节码；追加 var/venv-stage-a/Lib/site-packages 提供 API 依赖）
#
# 关键：显式使用本地 SQLite（沿用现有 var/dev-preview.db）与本地磁盘存储
# （var/local-storage）。即使 .env 从 .env.example 拷贝而带有 PostgreSQL/Redis/S3
# 默认值，本脚本仍强制本地化。由于 Windows 无法可靠表达“真·空字符串”环境变量
# （设为空=删除变量），我们不依赖父进程传播空值，而是通过 Python 引导脚本
# _local_boot.py 在目标进程内写入 os.environ（含 CELERY_BROKER_URL=''），
# pydantic-settings 会把空字符串当作真实值覆盖 .env 的 redis 默认。
#
# 幂等：重复启动不会产生重复进程。已经健康运行的 API/Web 会跳过（即使它们本
# 轮不是由本脚本启动，也绝不打扰/杀掉）；已存在的本地 Worker 会被识别并跳过。
# 只跟踪并（在 stop-local 中）停止本脚本真正启动的进程（记录 PID + 可执行路径 +
# 命令行标识，停止前逐一校验身份，避免误杀无关进程/复用 PID）。
#
# 用法：
#   start-local.cmd                     完整启动并打开浏览器
#   start-local.cmd -NoBrowser          启动但不打开浏览器
#   start-local.cmd -CheckOnly          只校验前置条件与当前状态，不启动任何服务
#   start-local.cmd -SkipWorker         跳过 Worker（仅前端/后端可用，案件会停在 queued）
#   start-local.cmd -SkipWeb -SkipWorker 仅启动 API
#
# 所有路径都由脚本所在目录（$PSScriptRoot）解析，不写死用户路径。
# =============================================================================

[CmdletBinding()]
param(
    [switch]$CheckOnly,   # 只做校验，不启动任何服务
    [switch]$NoBrowser,   # 不自动打开浏览器
    [switch]$SkipApi,     # 不启动/检测 API
    [switch]$SkipWeb,     # 不启动/检测 Web
    [switch]$SkipWorker   # 不启动/检测 Worker
)

$ErrorActionPreference = 'Stop'
$InformationPreference = 'Continue'

# ----------------------------------------------------------------------------
# 1. 路径解析（全部基于脚本所在目录）
# ----------------------------------------------------------------------------
$Root          = $PSScriptRoot                        # 仓库根
$Parent        = Split-Path $Root -Parent
$ApiPython     = Join-Path $Root 'var\venv-stage-a\Scripts\python.exe'
$ApiPyPath     = Join-Path $Root 'apps\api'
$WorkerPython  = Join-Path $Parent '项目\.venv\python.exe'
$WorkerSitepk  = Join-Path $Root 'var\venv-stage-a\Lib\site-packages'
$WebDir        = Join-Path $Root 'apps\web'
$WebIndex      = Join-Path $WebDir 'node_modules\vite\bin\vite.js'
$DbFile        = Join-Path $Root 'var\dev-preview.db'
$StorageDir    = Join-Path $Root 'var\local-storage'

$LogDir        = Join-Path $Root 'Cache\local-services'
$StateFile     = Join-Path $LogDir 'state.json'
$BootFile      = Join-Path $LogDir '_local_boot.py'

$ApiHost       = '127.0.0.1'
$ApiPort       = 8010
$WebHost       = '127.0.0.1'
$WebPort       = 5174
$ApiBase       = "http://${ApiHost}:${ApiPort}"
$WebBase       = "http://${WebHost}:${WebPort}"

$NodeExe       = $null
try { $NodeExe = (Get-Command node -ErrorAction Stop).Source } catch { $NodeExe = $null }

# ----------------------------------------------------------------------------
# 2. 小工具函数
# ----------------------------------------------------------------------------
function Write-Info([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "    [OK]  $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "    [!!]  $Message" -ForegroundColor Yellow
}

function Write-Err([string]$Message) {
    Write-Host "    [!!]  $Message" -ForegroundColor Red
}

# 以 .NET“进程环境块”写入环境变量（对非空值最稳；空值在 Windows 会被当作删除，故
# 本脚本不用它传播空字符串——真的空 broker 由 python 引导脚本在进程内设置）。
function Set-Env([string]$Name, [string]$Value) {
    [System.Environment]::SetEnvironmentVariable($Name, $Value, [System.EnvironmentVariableTarget]::Process)
}

function Test-Http([string]$Uri) {
    try {
        $r = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 3
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-ApiHealthy {
    try {
        $r = Invoke-WebRequest -Uri "$ApiBase/api/v1/health" -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -eq 200 -and $r.Content -match '"stage"\s*:\s*"A"')
    } catch {
        return $false
    }
}

function Test-WebHealthy {
    return (Test-Http "$WebBase/")
}

# 端口占用者 PID；无占用返回 $null（在受限环境可能查不到，作为尽力而为）
function Get-PortOwner([int]$Port) {
    try {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        if ($c) { return ($c | Select-Object -First 1).OwningProcess }
        return $null
    } catch {
        return $null
    }
}

# 是否已存在“本地轮询 Worker”进程（识别命令行里的 worker + 引导脚本/模块标记）
function Get-LocalWorkerPids {
    try {
        $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop
        $pids = @()
        foreach ($p in $procs) {
            $cmd = $p.CommandLine
            if ($cmd -and $cmd -match [regex]::Escape($Root) -and $cmd -match 'worker' -and $cmd -match '(_local_boot|drynaria_api\.cli)') {
                $pids += [int]$p.ProcessId
            }
        }
        return $pids
    } catch {
        throw 'Cannot inspect existing workers. Start from a normal Windows terminal.'
    }
}

function Read-State {
    if (Test-Path $StateFile) {
        try { return Get-Content $StateFile -Raw | ConvertFrom-Json } catch { return $null }
    }
    return $null
}

function Write-State($State) {
    $State | ConvertTo-Json -Depth 6 | Set-Content $StateFile -Encoding UTF8
}

# 校验某个“已记录的 owned 进程”是否仍然符合身份（防 PID 复用/误杀）
function Test-OwnedIdentity($ServiceRecord, [int]$ProcId) {
    if (-not $ServiceRecord) { return $false }
    if ($ProcId -le 0) { return $false }
    try {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction Stop
    } catch {
        return $false    # 进程不在了
    }
    if (-not $proc -or -not $proc.ExecutablePath -or -not $proc.CommandLine) { return $false }
    if ($proc.CommandLine -notmatch [regex]::Escape($Root)) { return $false }
    if ($ServiceRecord.executable -and $proc.ExecutablePath) {
        if ($proc.ExecutablePath -ne $ServiceRecord.executable) { return $false }
    }
    if ($ServiceRecord.cmd_marker -and $proc.CommandLine) {
        if ($proc.CommandLine -notmatch [regex]::Escape($ServiceRecord.cmd_marker)) { return $false }
    }
    return $true
}

function Wait-ApiAlive([int]$TimeoutSec = 45) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-ApiHealthy) { return $true }
        Start-Sleep -Milliseconds 700
    }
    return (Test-ApiHealthy)
}

# 向状态对象的 services 写入一个服务记录。PSCustomObject 不能通过点号新增属性，统一走
# Add-Member（属性已存在时直接赋值）。
function Set-ServiceState($State, [string]$Name, $Record) {
    $svc = $State.services
    if ($null -eq $svc) {
        $svc = [pscustomobject]@{}
        $State | Add-Member -NotePropertyName services -NotePropertyValue $svc -Force
    }
    $exists = $false
    foreach ($p in $svc.PSObject.Properties) { if ($p.Name -eq $Name) { $exists = $true; break } }
    if ($exists) { $svc.$Name = $Record }
    else { $svc | Add-Member -NotePropertyName $Name -NotePropertyValue $Record -Force }
}

function Wait-WebAlive([int]$TimeoutSec = 30) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-WebHealthy) { return $true }
        Start-Sleep -Milliseconds 600
    }
    return (Test-WebHealthy)
}

function Wait-ProcessAlive([int]$ProcId, [int]$TimeoutSec = 20) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Get-Process -Id $ProcId -ErrorAction SilentlyContinue) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return [bool](Get-Process -Id $ProcId -ErrorAction SilentlyContinue)
}

# ----------------------------------------------------------------------------
# 3. 前置校验
# ----------------------------------------------------------------------------
Write-Info "本地预览启动器（无 Docker）"
Write-Host "    仓库根: $Root"

$prereqOk = $true

if (-not (Test-Path $ApiPython)) {
    Write-Err "缺少 API 解释器: $ApiPython"
    Write-Err "   需先创建 var\venv-stage-a 虚拟环境（Python 3.11-3.12 + requirements）。"
    $prereqOk = $false
}
if (-not (Test-Path $WorkerPython)) {
    Write-Err "缺少 Worker 解释器: $WorkerPython"
    Write-Err "   仓库同级目录 项目\.venv 下应存在 python.exe。"
    $prereqOk = $false
}
if (-not (Test-Path (Join-Path $ApiPyPath 'drynaria_api\main.py'))) {
    Write-Err "缺少 API 源码: $(Join-Path $ApiPyPath 'drynaria_api\main.py')"
    $prereqOk = $false
}
if (-not (Test-Path $DbFile)) {
    Write-Warn "未找到现有 var\dev-preview.db：本次将指向该路径。"
    Write-Warn "   若为空库请先建表: var\venv-stage-a\Scripts\python.exe -m drynaria_api.cli schema-create"
}
if (-not $NodeExe) {
    Write-Err "未找到 node.exe（需要 Node.js 运行 Vite 前端）。"
    $prereqOk = $false
} elseif (-not (Test-Path $WebIndex)) {
    Write-Err "缺少 Vite 入口: $WebIndex"
    Write-Err "   请在 apps\web 目录先执行 npm install。"
    $prereqOk = $false
}

if (-not $prereqOk) {
    Write-Err "前置条件不满足，请先处理后再启动。"
    exit 1
}

try {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
} catch {
    Write-Err "无法创建日志目录: $LogDir"
    exit 1
}

# ----------------------------------------------------------------------------
# 4. 生成 python 引导脚本（Cache/local-services/_local_boot.py，自愈：每次写）
# ----------------------------------------------------------------------------
@'
# _local_boot.py — Windows 本地预览启动引导（由 start-local.ps1 生成/调用）
# 在 Python 进程内强制“本地只读预览”配置，摆脱对 Windows 环境变量空值传播的依赖。
# 用法：python _local_boot.py probe|uvicorn|worker

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # 仓库根
API_PATH = ROOT / "apps" / "api"
SITE_PACKAGES = ROOT / "var" / "venv-stage-a" / "Lib" / "site-packages"
STORAGE_DIR = ROOT / "var" / "local-storage"

API_HOST = "127.0.0.1"
API_PORT = 8010


def apply_local_env() -> None:
    os.environ["CELERY_BROKER_URL"] = ""
    os.environ["CELERY_RESULT_BACKEND"] = ""
    os.environ["REDIS_URL"] = ""
    os.environ["DATABASE_URL"] = "sqlite:///./var/dev-preview.db"
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["STORAGE_LOCAL_DIR"] = str(STORAGE_DIR)
    os.environ["ALLOWED_ORIGINS"] = "http://127.0.0.1:5174,http://localhost:5174"
    os.environ["AUTO_CREATE_TABLES"] = "false"
    os.environ["DRYNARIA_ENVIRONMENT"] = "dev"
    os.environ["WORKER_DEVICE"] = "cpu"
    os.environ["PYTHONUNBUFFERED"] = "1"

    sys.path.insert(0, str(API_PATH))
    if str(SITE_PACKAGES) not in sys.path:
        sys.path.append(str(SITE_PACKAGES))


def main(argv: list[str]) -> int:
    apply_local_env()
    target = argv[1] if len(argv) > 1 else "probe"

    if target == "probe":
        from drynaria_api.settings import Settings

        s = Settings()
        print("probe::database_url=" + repr(s.database_url))
        print("probe::storage_backend=" + repr(s.storage_backend))
        print("probe::storage_local_dir=" + repr(str(s.storage_local_dir)))
        print("probe::celery_broker_url=" + repr(s.celery_broker_url))
        print("probe::allowed_origins=" + repr(s.allowed_origins))
        print("probe::worker_guard_ok=" + repr(s.environment != "prod" and not s.celery_broker_url))
        print("probe::effective_storage=" + repr(s.effective_storage_backend))
        return 0

    if target == "uvicorn":
        import uvicorn

        uvicorn.run("drynaria_api.main:app", host=API_HOST, port=API_PORT, log_level="info")
        return 0

    if target == "worker":
        from drynaria_api.cli import main as cli_main

        return cli_main(["worker", "--poll-seconds", "2"])

    print("local-boot: unknown target " + repr(target))
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'@ | Set-Content -Path $BootFile -Encoding UTF8

Write-Info "强制本地化（由引导脚本在进程内写入）"
Write-Ok "DATABASE_URL       = sqlite:///./var/dev-preview.db (沿用现有库)"
Write-Ok "STORAGE_BACKEND    = local (本地磁盘 var/local-storage)"
Write-Ok "CELERY_BROKER_URL  = <空> (覆盖 .env 中 redis 默认；在 python 进程内设置)"
Write-Ok "ALLOWED_ORIGINS    = http://127.0.0.1:$WebPort,http://localhost:$WebPort"

# ----------------------------------------------------------------------------
# 5. 配置解析自检（probe）——任何模式都跑一次，证明“空 broker”确实生效
# ----------------------------------------------------------------------------
Push-Location $Root
$probeOut = @(& $ApiPython $BootFile probe)
$probeExit = $LASTEXITCODE
Pop-Location

$ok = $true
Write-Info "配置解析自检（空 broker -> 保留空字符串）"
foreach ($line in $probeOut) {
    if ($line -match '^probe::(.*)$') {
        Write-Host "    $($Matches[1])"
    }
    if ($line -match '^probe::celery_broker_url=(.+)$') {
        if ($Matches[1] -ne "''") { $ok = $false }
    }
}
if ($probeExit -ne 0) { $ok = $false; Write-Err "配置文件解析进程以退出码 $probeExit 结束。" }
if ($ok) { Write-Ok "空 CELERY_BROKER_URL 校验通过（已覆盖 .env 的 redis 默认）" }
else { Write-Err "空 CELERY_BROKER_URL 未生效，请检查引导脚本。" }

if ($CheckOnly) {
    Write-Info "当前状态（不启动）"
    Write-Host "    API:    $(if (Test-ApiHealthy) {'healthy'} elseif (Get-PortOwner $ApiPort) {"端口 $ApiPort 被 PID $(Get-PortOwner $ApiPort) 占用(非健康)"} else {'空闲, 将启动'})"
    Write-Host "    Web:    $(if (Test-WebHealthy) {'healthy'} elseif (Get-PortOwner $WebPort) {"端口 $WebPort 被 PID $(Get-PortOwner $WebPort) 占用(非健康)"} else {'空闲, 将启动'})"
    $wps = Get-LocalWorkerPids
    if ($wps.Count -gt 0) { Write-Host "    Worker: 已存在本地 Worker PID $($wps -join ', ') (将跳过启动)" }
    elseif (-not (Test-Path $WorkerPython)) { Write-Host "    Worker: 解释器缺失" }
    else { Write-Host "    Worker: 空闲, 将启动" }
    Write-Ok "CheckOnly 完成，未启动任何服务。"
    exit 0
}

# ----------------------------------------------------------------------------
# 6. 读取状态，开始启动各服务
# ----------------------------------------------------------------------------
$state = Read-State
if (-not $state) {
    $state = [pscustomobject]@{ schema_version = 1; signature = 'drynaria-local-services-v1'; root = $Root; services = [pscustomobject]@{} }
}
if (-not $state.services) { $state | Add-Member -NotePropertyName services -NotePropertyValue ([pscustomobject]@{}) -Force }

$startedOrPresent = @()

# ---------------------------------------------------------------------------
# 6.1 API
# ---------------------------------------------------------------------------
if (-not $SkipApi) {
    $apiRecord = $state.services.api
    if (Test-ApiHealthy) {
        if ($apiRecord -and (Test-OwnedIdentity $apiRecord ([int]$apiRecord.pid))) {
            Write-Info "API 已在运行（本启动器所启动，PID $($apiRecord.pid)），跳过启动。"
        } else {
            Write-Warn "检测到健康 API 已在 $ApiBase 运行（非本启动器启动），跳过启动且不会被 stop-local 停止。"
        }
        $startedOrPresent += 'api'
    } else {
        $portOwner = Get-PortOwner $ApiPort
        if ($portOwner) {
            Write-Err "端口 $ApiPort 已被 PID $portOwner 占用且健康检查失败："
            Write-Err "   该进程并非可用的阶段 A API，无法启动本服务（不会杀掉它）。"
            exit 1
        }
        $apiOut = Join-Path $LogDir 'api.out.log'
        $apiErr = Join-Path $LogDir 'api.err.log'
        $apiP = Start-Process -FilePath $ApiPython `
            -ArgumentList @($BootFile, 'uvicorn') `
            -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $apiOut -RedirectStandardError $apiErr
        Write-Info "已启动 API（PID $($apiP.Id)）→ 等待健康…"
        if (Wait-ApiAlive) {
            Write-Ok "API 健康: $($ApiBase)/api/v1/health"
            Set-ServiceState $state 'api' ([pscustomobject]@{
                pid        = $apiP.Id
                executable = $ApiPython
                cmd_marker = 'uvicorn'
                started_at = (Get-Date).ToString('o')
            })
            $startedOrPresent += 'api'
        } else {
            Write-Err "API 未能在超时内健康，请查看 $apiErr"
            exit 1
        }
    }
} else {
    Write-Warn "已跳过 API（-SkipApi）。前端与 Worker 将无法正常分析。"
}

# ---------------------------------------------------------------------------
# 6.2 Web（Vite）
# ---------------------------------------------------------------------------
if (-not $SkipWeb) {
    $webRecord = $state.services.web
    if (Test-WebHealthy) {
        if ($webRecord -and (Test-OwnedIdentity $webRecord ([int]$webRecord.pid))) {
            Write-Info "Web 已在运行（本启动器所启动，PID $($webRecord.pid)），跳过启动。"
        } else {
            Write-Warn "检测到健康 Web 已在 $WebBase 运行（非本启动器启动），跳过启动且不会被 stop-local 停止。"
        }
        $startedOrPresent += 'web'
    } else {
        $webOwner = Get-PortOwner $WebPort
        if ($webOwner) {
            Write-Err "端口 $WebPort 已被 PID $webOwner 占用且健康检查失败，无法启动（不会杀掉它）。"
            exit 1
        }
        Set-Env 'VITE_API_PROXY' $ApiBase
        $webOut = Join-Path $LogDir 'web.out.log'
        $webErr = Join-Path $LogDir 'web.err.log'
        $webP = Start-Process -FilePath $NodeExe `
            -ArgumentList @($WebIndex,'--host',$WebHost,'--port',"$WebPort",'--strictPort') `
            -WorkingDirectory $WebDir -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $webOut -RedirectStandardError $webErr
        Write-Info "已启动 Web（PID $($webP.Id)）→ 等待响应…"
        if (Wait-WebAlive) {
            Write-Ok "Web 已可用: $WebBase  =>  /api 代理到 $ApiBase"
            Set-ServiceState $state 'web' ([pscustomobject]@{
                pid        = $webP.Id
                executable = $NodeExe
                cmd_marker = 'vite'
                started_at = (Get-Date).ToString('o')
            })
            $startedOrPresent += 'web'
        } else {
            Write-Err "Web 未能在超时内就绪，请查看 $webErr"
            exit 1
        }
    }
} else {
    Write-Warn "已跳过 Web（-SkipWeb）。"
}

# ---------------------------------------------------------------------------
# 6.3 Worker（本地轮询，真实模型）
# ---------------------------------------------------------------------------
if (-not $SkipWorker) {
    $workerRecord = $state.services.worker
    $existingWorkers = Get-LocalWorkerPids
    if ($existingWorkers.Count -gt 0) {
        $hers = @($existingWorkers | Where-Object { -not ($workerRecord -and $_ -eq [int]$workerRecord.pid) })
        if ($hers.Count -gt 0) {
            Write-Warn "检测到已存在的本地 Worker（PID $($hers -join ', ')），跳过启动以避免重复消费。"
            $startedOrPresent += 'worker'
        } else {
            if ($workerRecord -and (Test-OwnedIdentity $workerRecord ([int]$workerRecord.pid))) {
                Write-Info "Worker 已在运行（本启动器所启动，PID $($workerRecord.pid)），跳过启动。"
            } else {
                Write-Info "Worker 进程存在（PID $($existingWorkers -join ', ')），跳过启动。"
            }
            $startedOrPresent += 'worker'
        }
    } else {
        $workerOut = Join-Path $LogDir 'worker.out.log'
        $workerErr = Join-Path $LogDir 'worker.err.log'
        $wP = Start-Process -FilePath $WorkerPython `
            -ArgumentList @('-B', $BootFile, 'worker') `
            -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $workerOut -RedirectStandardError $workerErr
        Write-Info "已启动 Worker（PID $($wP.Id)）→ 等待进程就绪…"
        Start-Sleep -Seconds 3
        if (Wait-ProcessAlive $wP.Id 15) {
            Write-Ok "Worker 进程运行中（PID $($wP.Id)）。日志: $workerOut"
            if (Test-Path $workerErr) { Write-Warn "Worker stderr: $workerErr" }
            Set-ServiceState $state 'worker' ([pscustomobject]@{
                pid        = $wP.Id
                executable = $WorkerPython
                cmd_marker = '_local_boot'
                started_at = (Get-Date).ToString('o')
            })
            $startedOrPresent += 'worker'
        } else {
            Write-Err "Worker 未能在超时内保持运行，请查看 $workerErr"
            exit 1
        }
    }
} else {
    Write-Warn "已跳过 Worker（-SkipWorker）。案件会一直停在 queued，需手动运行 Worker。"
}

# ---------------------------------------------------------------------------
# 7. 保存状态、汇总、开浏览器
# ---------------------------------------------------------------------------
try { Write-State $state } catch { Write-Warn "无法写入状态文件: $StateFile" }

Write-Info "启动汇总"
Write-Host "    API:    $ApiBase  (127.0.0.1:$ApiPort)"
Write-Host "    Web:    $WebBase  (127.0.0.1:$WebPort)"
Write-Host "    Worker: 后台轮询（真实模型）"
Write-Host "    日志:   $LogDir"

if ($startedOrPresent -contains 'web' -and -not $NoBrowser) {
    try {
        Write-Info "打开浏览器: $WebBase"
        Start-Process "$WebBase"
    } catch {
        Write-Warn "无法打开浏览器（可重跑加 -NoBrowser）。"
    }
}

Write-Ok "本地预览已就绪：$WebBase"
exit 0
