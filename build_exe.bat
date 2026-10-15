@echo off
setlocal
chcp 65001 >nul
title IVAS Portable EXE Builder

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_portable_exe.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"

if not "%BUILD_EXIT%"=="0" (
    echo.
    echo [FAILED] IVAS portable build failed with exit code %BUILD_EXIT%.
) else (
    echo.
    echo [OK] Portable application is in "%~dp0build\IVAS".
)

if not defined CI pause
exit /b %BUILD_EXIT%
