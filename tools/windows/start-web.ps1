param(
    [ValidateRange(1, 65535)][int]$Port = 60550,
    [Alias('NoBrowser')][switch]$ServerOnly,
    [ValidatePattern('^Local\\FlowHMR-backend-[0-9a-f]{32}$')][string]$DesktopJobName
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $projectRoot
. (Join-Path $PSScriptRoot 'desktop-runtime.ps1')
if ($DesktopJobName) {
    if (-not $ServerOnly) { throw 'A desktop job can only start the server.' }
    Join-FlowHMRDesktopJob $DesktopJobName
}
$pythonExe = Get-FlowHMRConsoleRuntime $projectRoot
if (-not $ServerOnly) {
    $launcher = Join-Path $projectRoot 'FlowHMR.exe'
    if (Test-Path -LiteralPath $launcher -PathType Leaf) {
        Start-Process -FilePath $launcher -ArgumentList @('--port',"$Port") -WorkingDirectory $projectRoot -WindowStyle Normal
        exit 0
    }
    $desktopExe = Get-FlowHMRDesktopRuntime $projectRoot
    Start-Process -FilePath $desktopExe -ArgumentList @('-I','tools\windows\desktop_launcher.pyw','--port',"$Port") -WorkingDirectory $projectRoot -WindowStyle Normal
    exit 0
}
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$env:FLOWHMR_SAM_COMPILE = '0'
$env:FLOWHMR_ATTENTION_BACKEND = 'query_batched'
$env:FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE = '256'
$url = "http://127.0.0.1:$Port"
$logDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stateFile = Join-Path $logDir 'web-process.json'
$serverProcess = $null
if (Test-Path -LiteralPath $stateFile) {
    try {
        $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        $recordedId = [int]$state.pid
        $recordedPort = [int]$state.port
        $recordedKey = [string]$state.key
    } catch {
        Write-Warning 'The saved FlowHMR process record is invalid.'
        $state = $null
    }
    if ($state.root -eq $projectRoot -and $recordedId -gt 0 -and $recordedPort -ge 1 -and $recordedPort -le 65535) {
        $running = Get-Process -Id $recordedId -ErrorAction SilentlyContinue
        if ($running -and $running.Path -eq $pythonExe) {
            try {
                $recorded = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedId"
            } catch {
                if ($DesktopJobName) {
                    Write-Warning 'Could not verify the recorded FlowHMR service.'
                    exit 20
                }
                throw
            }
            if ($recorded.ExecutablePath -eq $pythonExe -and $recorded.CommandLine -match "\bapp\.py\b.*--host\s+127\.0\.0\.1\s+--port\s+$recordedPort\b") {
                $serverProcess = $running
            }
        }
    }
}

if ($DesktopJobName -and $serverProcess) {
    # The native character picker authenticates against the service started
    # by this window. A surviving service may also have older API routes.
    try {
        if ($recordedPort -ne $Port) {
            $oldWindow = $null
            if ([System.Threading.Mutex]::TryOpenExisting("Local\FlowHMR-desktop-$recordedPort", [ref]$oldWindow)) {
                $oldWindow.Dispose()
                throw "A FlowHMR window is still open on port $recordedPort."
            }
        }
        $oldUrl = "http://127.0.0.1:$recordedPort"
        $oldInfo = Invoke-RestMethod -Uri "$oldUrl/api/info" -UseBasicParsing -TimeoutSec 5
        $oldJobs = Invoke-RestMethod -Uri "$oldUrl/api/jobs?limit=1" -UseBasicParsing -TimeoutSec 5
        $oldRuntime = Invoke-RestMethod -Uri "$oldUrl/api/runtime" -UseBasicParsing -TimeoutSec 8
        if ($oldInfo.model_files.directory -ne (Join-Path $projectRoot 'models') -or
            $null -eq $oldJobs.total -or $null -eq $oldJobs.deletable_count) {
            throw 'Could not verify the state of the previous FlowHMR service.'
        }
        # A surviving service was started with an older FLOWHMR_DESKTOP_CHARACTER_KEY, so
        # the native character picker in this window would get 403 from /api/desktop/*.
        # Restarting is the only way to resync the key; only do it when nothing is running.
        $keyChanged = $recordedKey -and ($recordedKey -ne $env:FLOWHMR_DESKTOP_CHARACTER_KEY)
        $busy = $oldInfo.model_files.download.status -in @('running', 'stopping') -or
                $oldInfo.model_files.import.state -eq 'running' -or
                $oldRuntime.download.status -in @('running', 'stopping')
        if ($keyChanged -and -not $busy) {
            Write-Host 'Desktop key changed; restarting the service so the character picker stays usable.'
        } elseif ($oldJobs.total -ne $oldJobs.deletable_count -or $busy) {
            throw 'The previous FlowHMR service still has work in progress.'
        }
        & (Join-Path $PSScriptRoot 'stop-web.ps1') -ExpectedPid $recordedId
        if (-not $serverProcess.WaitForExit(3000)) { throw 'The previous FlowHMR service did not exit.' }
        $serverProcess = $null
    } catch {
        Write-Warning $_.Exception.Message
        exit 20
    }
} elseif ($serverProcess -and $recordedPort -ne $Port) {
    throw "FlowHMR is already running on port $recordedPort. Stop it before changing the port."
}

if ($null -eq $serverProcess) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try { $listener.Start() } catch { throw "Port $Port is occupied. Choose another port with -Port or stop the existing server." } finally { $listener.Stop() }
    $serverProcess = Start-Process -FilePath $pythonExe -ArgumentList @('-u','app.py','--host','127.0.0.1','--port',"$Port",'--device','cuda','--sam-chunk-default','24') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'web.stdout.log') -RedirectStandardError (Join-Path $logDir 'web.stderr.log') -PassThru
    @{pid=$serverProcess.Id;port=$Port;root=$projectRoot;key=$env:FLOWHMR_DESKTOP_CHARACTER_KEY} | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding UTF8
    Write-Host "Starting FlowHMR. Logs: $logDir"
}
$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
    if ($serverProcess.HasExited) {
        Get-Content -LiteralPath (Join-Path $logDir 'web.stderr.log') -Tail 40
        throw 'FlowHMR exited during startup.'
    }
    try {
        if ($DesktopJobName) {
            $headers = @{ 'X-FlowHMR-Desktop-Key' = $env:FLOWHMR_DESKTOP_CHARACTER_KEY }
            $response = Invoke-WebRequest -Uri "$url/api/desktop/session" -UseBasicParsing -TimeoutSec 2 -Headers $headers
            $ready = $response.StatusCode -eq 204
        } else {
            $response = Invoke-WebRequest -Uri "$url/api/info" -UseBasicParsing -TimeoutSec 2
            $ready = $response.StatusCode -eq 200
        }
        if ($ready) {
            Write-Host "FlowHMR ready: $url"
            exit 0
        }
    } catch {}
    Start-Sleep -Milliseconds 500
}
throw "Startup is still running. Inspect $logDir or run stop-web.cmd."
