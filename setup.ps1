# One-time setup for Windows WITHOUT admin rights. Double-click setup.bat, or run:
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1 [-ModelsDir D:\models] [-NoModels] [-SkipSelftest]
# Everything is downloaded into .\tools and .\.venv next to this script (portable, nothing is installed system-wide).
# Written for the built-in Windows PowerShell 5.1.
param([string]$ModelsDir = "", [switch]$NoModels, [switch]$SkipSelftest)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest ~10x faster
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$UV_VER   = "0.13.1"
$NODE_VER = "v24.22.0"
$GIT_TAG  = "v2.56.0.windows.2"
$GIT_VER  = "2.56.0.2"
$LLAMA    = "b11700"
$PI_VER   = "1.3.0"
$TORCH    = "2.14.1"

$Root  = $PSScriptRoot
$Tools = Join-Path $Root "tools"
$Dl    = Join-Path $Tools "_downloads"
New-Item -ItemType Directory -Force $Tools, $Dl, (Join-Path $Root "state") | Out-Null

function Say($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Get-File($url, $name) {
    $out = Join-Path $Dl $name
    if (-not (Test-Path $out)) {
        Write-Host "    downloading $url"
        Invoke-WebRequest -Uri $url -OutFile "$out.part" -UseBasicParsing
        Move-Item "$out.part" $out
    }
    return $out
}
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }
function Expand-To($zip, $dest) {
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Expand-Archive -Path $zip -DestinationPath $dest -Force
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
$env:UV_CACHE_DIR          = Join-Path $Tools "uv-cache"
$env:UV_LINK_MODE          = "copy"        # exFAT has no hard links
$env:HF_HOME               = Join-Path $Tools "hf-home"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUTF8 = "1"

# ---------------------------------------------------------------- 1. uv + Python
Say "1/7 uv (Python manager) and Python 3.12"
$uv = Join-Path $Tools "uv\uv.exe"
if (-not (Test-Path $uv)) {
    $z = Get-File "https://github.com/astral-sh/uv/releases/download/$UV_VER/uv-x86_64-pc-windows-msvc.zip" "uv.zip"
    Expand-To $z (Join-Path $Tools "uv")
}
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { & $uv venv (Join-Path $Root ".venv") --python 3.12 --managed-python; Check "creating the Python environment" }

# ---------------------------------------------------------------- 2. Python packages
Say "2/7 Python packages (torch picks CPU or CUDA automatically)"
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$hasNvidia = [bool]$nvsmi
if ($hasNvidia) { & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader }
else { Write-Host "    no NVIDIA GPU found -> CPU build" }
& $uv pip install --python $py "torch==$TORCH" --torch-backend auto; Check "installing torch"
& $uv pip install --python $py -r (Join-Path $Root "requirements.txt"); Check "installing Python packages"
if ($hasNvidia) {
    & $uv pip install --python $py "bitsandbytes==0.50.2"; Check "installing bitsandbytes"
    # optional fast kernels for the Gated-DeltaNet layers; training works without them, just slower
    & $uv pip install --python $py triton-windows flash-linear-attention
    if ($LASTEXITCODE -ne 0) { Write-Host "    (optional fast kernels skipped)" }
}

$env:PYTHONPATH = Join-Path $Root "src"
& $py -m adv.hardware --write (Join-Path $Root "state\hardware.json"); Check "hardware detection"
$hwProfile = (Get-Content (Join-Path $Root "state\hardware.json") -Raw | ConvertFrom-Json).profile
Write-Host "    hardware profile: $hwProfile" -ForegroundColor Green

# ---------------------------------------------------------------- 3. Node.js + Pi
Say "3/7 Node.js $NODE_VER and the Pi coding agent $PI_VER"
$nodeDir = Join-Path $Tools "node"
if (-not (Test-Path (Join-Path $nodeDir "node.exe"))) {
    $z = Get-File "https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-win-x64.zip" "node.zip"
    Expand-To $z (Join-Path $Tools "_node")
    Move-Item (Join-Path $Tools "_node\node-$NODE_VER-win-x64") $nodeDir
    Remove-Item -Recurse -Force (Join-Path $Tools "_node")
}
$env:PATH = "$nodeDir;$env:PATH"
$env:npm_config_cache = Join-Path $Tools "npm-cache"
& (Join-Path $nodeDir "npm.cmd") install --prefix (Join-Path $Tools "pi") --no-fund --no-audit "@earendil-works/pi-coding-agent@$PI_VER"
Check "installing Pi"

# ---------------------------------------------------------------- 4. Git Bash (Pi's bash tool)
Say "4/7 PortableGit (gives Pi a bash shell)"
if (-not (Test-Path (Join-Path $Tools "git\bin\bash.exe"))) {
    $sfx = Get-File "https://github.com/git-for-windows/git/releases/download/$GIT_TAG/PortableGit-$GIT_VER-64-bit.7z.exe" "PortableGit.7z.exe"
    Start-Process -FilePath $sfx -ArgumentList "-o`"$(Join-Path $Tools 'git')`"", "-y" -Wait -NoNewWindow
}
if (-not (Test-Path (Join-Path $Tools "git\bin\bash.exe"))) { throw "PortableGit did not unpack (tools\git\bin\bash.exe missing)" }

# ---------------------------------------------------------------- 5. llama.cpp
Say "5/7 llama.cpp $LLAMA ($(if ($hasNvidia) {'CUDA 12.4'} else {'CPU'}) build) + LoRA converter"
$llamaDir = Join-Path $Tools "llama"
if (-not (Get-ChildItem -Path $llamaDir -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue)) {
    if ($hasNvidia) {
        $z = Get-File "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA/llama-$LLAMA-bin-win-cuda-12.4-x64.zip" "llama.zip"
        $c = Get-File "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA/cudart-llama-bin-win-cuda-12.4-x64.zip" "cudart.zip"
        Expand-To $z $llamaDir
        Expand-Archive -Path $c -DestinationPath $llamaDir -Force
    } else {
        $z = Get-File "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA/llama-$LLAMA-bin-win-cpu-x64.zip" "llama.zip"
        Expand-To $z $llamaDir
    }
}
$srcDir = Join-Path $Tools "llama.cpp-src"
if (-not (Get-ChildItem -Path $srcDir -Recurse -Filter "convert_lora_to_gguf.py" -ErrorAction SilentlyContinue)) {
    $z = Get-File "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LLAMA.zip" "llama-src.zip"
    New-Item -ItemType Directory -Force $srcDir | Out-Null
    $p = "llama.cpp-$LLAMA"
    $ok = $false
    try { & tar.exe -xf $z -C $srcDir "$p/convert_lora_to_gguf.py" "$p/convert_hf_to_gguf.py" "$p/gguf-py" "$p/conversion"; $ok = ($LASTEXITCODE -eq 0) } catch { $ok = $false }
    if (-not $ok) { Expand-To $z $srcDir }   # slower: unpacks the whole source tree
}

# ---------------------------------------------------------------- 6. models
if (-not $NoModels) {
    Say "6/7 models for profile '$hwProfile'"
    if ($ModelsDir -eq "") {
        $default = Join-Path $Root "models"
        Write-Host "    The cpu profiles fit on the SD card. The 27B profile needs ~76 GB, so use the PC's own drive"
        Write-Host "    (note: some lab PCs wipe user folders at logout)."
        $ans = Read-Host "    Folder for models (Enter = $default)"
        if ($ans -ne "") { $ModelsDir = $ans } else { $ModelsDir = $default }
    }
    & $py -m adv.download --profile $hwProfile --models-dir $ModelsDir --yes
    if ($LASTEXITCODE -ne 0) { throw "model download failed" }
} else { Say "6/7 models skipped (-NoModels)" }

# ---------------------------------------------------------------- 7. self-test
if (-not $SkipSelftest -and -not $NoModels) {
    Say "7/7 self-test: can llama.cpp serve our LoRA adapters on this machine? (and d3-flash)"
    & $py -m adv.selftest --profile $hwProfile
} else { Say "7/7 self-test skipped" }

Remove-Item -Recurse -Force $Dl -ErrorAction SilentlyContinue
& $uv cache clean | Out-Null   # the installed packages are copies; the cache only wastes card space
Say "Setup finished. Next:  dryrun.bat (2 minutes, fake model)  then  run.bat  (real training)"
