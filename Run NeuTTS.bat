@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  NeuTTS-AIR Launcher
REM ─────────────────────────────────────────────────────────────────────────────

REM Set Hugging Face cache directory to local drive to avoid C: drive space issues
set HF_HOME=%~dp0\.hf_cache
set HUGGINGFACE_HUB_CACHE=%~dp0\.hf_cache\hub

REM Create cache directories if they don't exist
if not exist "%~dp0\.hf_cache" mkdir "%~dp0\.hf_cache"
if not exist "%~dp0\.hf_cache\hub" mkdir "%~dp0\.hf_cache\hub"

REM Configure PyTorch CUDA memory allocator to reduce fragmentation
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Suppress Hugging Face symlink warnings on Windows
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

REM ─────────────────────────────────────────────────────────────────────────────
REM  Kill any leftover NeuTTS / Gradio process on ports 7860-7900
REM ─────────────────────────────────────────────────────────────────────────────
echo Checking for existing NeuTTS instances...
for /L %%P in (7860,1,7900) do (
    for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /R ":%%P.*LISTENING"') do (
        echo   Killing process %%a on port %%P
        taskkill /PID %%a /F >nul 2>&1
    )
)
echo Done.
echo.

echo Hugging Face cache : %HF_HOME%
echo CUDA alloc conf    : %PYTORCH_CUDA_ALLOC_CONF%
echo.

REM ─────────────────────────────────────────────────────────────────────────────
REM  Launch app
REM ─────────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"
.venv\python.exe app.py
pause
