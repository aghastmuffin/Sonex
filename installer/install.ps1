# Sonex installer for Windows (PowerShell)
#
#   irm https://raw.githubusercontent.com/aghastmuffin/Sonex/main/installer/install.ps1 | iex
#
# Options (env vars):
#   $env:SONEX_HOME = "C:\Users\you\Sonex"
#   $env:SONEX_REF = "0.4-b"
#   $env:SONEX_WITH_MFA = "1"

$ErrorActionPreference = "Stop"

$Owner = "aghastmuffin"
$Repo = "Sonex"
$Ref = if ($env:SONEX_REF) { $env:SONEX_REF } else { "main" }
$RawBase = "https://raw.githubusercontent.com/$Owner/$Repo/$Ref/installer"

Write-Host "==> Sonex installer"
Write-Host "    ref: $Ref"

function Find-Python {
    $candidates = @(
        "py -3.12", "py -3.11", "py -3.10", "py -3.9",
        "python3.12", "python3.11", "python3.10", "python3.9",
        "python3", "python"
    )
    foreach ($cmd in $candidates) {
        try {
            $parts = $cmd -split " ", 2
            $exe = $parts[0]
            $arg = if ($parts.Length -gt 1) { $parts[1] } else { $null }
            $args = @()
            if ($arg) { $args += $arg }
            $args += "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
            $ver = & $exe @args 2>$null
            if ($LASTEXITCODE -eq 0 -or $ver) {
                $major, $minor = $ver.Trim().Split(".")
                if ([int]$major -gt 3 -or ([int]$major -eq 3 -and [int]$minor -ge 9)) {
                    return @{ Exe = $exe; Arg = $arg }
                }
            }
        } catch {}
    }
    Write-Error "Python 3.9+ is required. Install from https://www.python.org/downloads/"
}

$py = Find-Python
$pythonCmd = if ($py.Arg) { "$($py.Exe) $($py.Arg)" } else { $py.Exe }
Write-Host "    python: $pythonCmd"

$tmp = Join-Path $env:TEMP ("sonex-install-" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmp | Out-Null
$bootstrap = Join-Path $tmp "bootstrap.py"

Invoke-WebRequest -Uri "$RawBase/bootstrap.py" -OutFile $bootstrap -UseBasicParsing

$bootstrapArgs = @("--ref", $Ref)
if ($env:SONEX_HOME) { $bootstrapArgs += @("--dir", $env:SONEX_HOME) }
if ($env:SONEX_WITH_MFA -eq "1") { $bootstrapArgs += "--with-mfa" }

if ($py.Arg) {
    & $py.Exe $py.Arg $bootstrap @bootstrapArgs
} else {
    & $py.Exe $bootstrap @bootstrapArgs
}

exit $LASTEXITCODE
