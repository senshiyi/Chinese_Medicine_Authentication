@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start-local.ps1" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
