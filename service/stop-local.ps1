# =============================================================================
# stop-local.ps1 — 停止本地预览的“已归属”进程
#
# 安全原则：
#   * 只停止由 start-local.ps1 记录在 Cache\local-services\state.json 中的进程。
#   * 每个 PID 在停止前都用“可执行路径 + 命令行标记”做身份校验，校验不过
#     （进程已退出 / PID 被复用 / 不是我们启动的）就不执行 kill，避免误杀。
#   * 由其它来源（如 Codex / 手动）启动的 API/Web/Worker 一律不触碰。
#
# 用法：
#   stop-local.cmd                 停止全部已归属服务
#   stop-local.cmd -Api            只停 API
#   stop-local.cmd -Web            只停 Web
#   stop-local.cmd -Worker         只停 Worker
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Api,
    [switch]$Web,
    [switch]$Worker,
    [switch]$All
)

$ErrorActionPreference = 'Stop'

$Root      = $PSScriptRoot
$LogDir    = Join-Path $Root 'Cache\local-services'
$StateFile = Join-Path $LogDir 'state.json'

function Write-Info([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message)   { Write-Host "    [OK]  $Message" -ForegroundColor Green }
function Write-Warn([string]$Message) { Write-Host "    [!!]  $Message" -ForegroundColor Yellow }

function Read-State {
    if (Test-Path $StateFile) {
        try { return Get-Content $StateFile -Raw | ConvertFrom-Json } catch { return $null }
    }
    return $null
}

function Write-State($State) {
    $State | ConvertTo-Json -Depth 6 | Set-Content $StateFile -Encoding UTF8
}

function Test-OwnedIdentity($ServiceRecord, [int]$ProcId) {
    if (-not $ServiceRecord) { return $false }
    if ($ProcId -le 0) { return $false }
    try {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction Stop
    } catch {
        return $false
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

Write-Info "停止本地预览（仅归属进程）"

$state = Read-State
if (-not $state -or -not $state.services) {
    Write-Warn "没有可停止的已归属服务（未找到 $StateFile 或为空）。"
    Write-Warn "其它来源(如 Codex)启动的进程不会被本脚本停止。"
    exit 0
}

# 默认（不带任何服务开关）＝停止全部；否则只停指定项
$wantAll = $All -or (-not $Api -and -not $Web -and -not $Worker)
$services = $state.services
$anyStopped = $false

foreach ($name in @('api', 'web', 'worker')) {
    $rec = $services.$name
    if (-not $rec) { continue }
    if (-not $wantAll) {
        if ($name -eq 'api'    -and -not $Api)    { continue }
        if ($name -eq 'web'    -and -not $Web)    { continue }
        if ($name -eq 'worker' -and -not $Worker) { continue }
    }

    $pidVal = [int]$rec.pid
    if ($pidVal -le 0) {
        Write-Warn "$name 记录中的 PID 无效，跳过。"
        $services.PSObject.Properties.Remove($name)
        continue
    }

    if (Test-OwnedIdentity $rec $pidVal) {
        try {
            Stop-Process -Id $pidVal -Force -ErrorAction Stop
            Write-Ok "已停止 $name（PID $pidVal）"
            $anyStopped = $true
        } catch {
            Write-Warn "停止 $name（PID $pidVal）失败：$($_.Exception.Message)"
        }
    } else {
        Write-Warn "$name（PID $pidVal）身份校验未通过（进程已退出或 PID 被复用），未执行 kill。"
    }
    # 无论是否成功停止，都从记录中移除（避免下次重复处理）
    $services.PSObject.Properties.Remove($name)
}

if ($anyStopped) {
    Write-Info "更新状态文件…"
    try {
        Write-State $state
        Write-Ok "已更新 $StateFile"
    } catch {
        Write-Warn "无法更新状态文件：$($_.Exception.Message)"
    }
} else {
    Write-Warn "没有任何已归属进程被停止。"
}

Write-Warn "提示：非本脚本启动的进程（如 Codex 的 API/Web/Worker）不受影响。"
exit 0
