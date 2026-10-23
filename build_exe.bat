@echo off
setlocal
chcp 65001 >nul
title IVAS Portable EXE Builder

if not exist "%~dp0build_portable_exe.ps1" (
    echo [FAILED] Missing build_portable_exe.ps1 beside this launcher.
    if not defined CI pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_portable_exe.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"

if not "%BUILD_EXIT%"=="0" (
    echo.
    echo [FAILED] IVAS portable build failed with exit code %BUILD_EXIT%.
    echo Build log: "%~dp0tmp\portable-build.log"
) else (
    echo.
    echo [OK] Portable application is in "%~dp0build\IVAS".
    echo Run "%~dp0build\IVAS\IVAS.exe".
)

if not defined CI pause
exit /b %BUILD_EXIT%
