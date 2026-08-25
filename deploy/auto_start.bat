@echo off
REM ============================================================
REM Trading Bot Auto-Start for Windows
REM ============================================================
REM This script:
REM   1. Creates a Windows scheduled task to start the bot on boot
REM   2. Runs the bot in background (no console window)
REM   3. Auto-restarts if the bot crashes
REM
REM Usage: Run this once as Administrator
REM ============================================================

echo Setting up Trading Bot auto-start...
echo.

REM Get the Python path
set PYTHON_PATH=D:\Program Files\Python\python.exe
set PROJECT_PATH=%~dp0..
set BOT_SCRIPT=%PROJECT_PATH%\scripts\run_github_bot.py
set LOG_FILE=%PROJECT_PATH%\data\results\bot_output.log
set WATCHDOG=%PROJECT_PATH%\deploy\watchdog.py

REM Create the watchdog script (restarts bot if it crashes)
echo import subprocess, time, sys > "%WATCHDOG%"
echo. >> "%WATCHDOG%"
echo BOT_SCRIPT = r"%BOT_SCRIPT%" >> "%WATCHDOG%"
echo PYTHON = r"%PYTHON_PATH%" >> "%WATCHDOG%"
echo LOG = r"%LOG_FILE%" >> "%WATCHDOG%"
echo. >> "%WATCHDOG%"
echo while True: >> "%WATCHDOG%"
echo     with open(LOG, "a") as log: >> "%WATCHDOG%"
echo         print(f"[WATCHDOG] Starting bot at {__import__('datetime').datetime.now()}", file=log) >> "%WATCHDOG%"
echo         proc = subprocess.Popen([PYTHON, "-u", BOT_SCRIPT], stdout=log, stderr=log) >> "%WATCHDOG%"
echo         proc.wait() >> "%WATCHDOG%"
echo         print(f"[WATCHDOG] Bot exited with code {proc.returncode}. Restarting in 30s...", file=log) >> "%WATCHDOG%"
echo     time.sleep(30) >> "%WATCHDOG%"

REM Create scheduled task
echo Creating scheduled task...
schtasks /create /tn "TradingBot" /tr "pythonw.exe \"%WATCHDOG%\"" /sc onstart /ru %USERNAME% /rl highest /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! Trading bot will auto-start on boot.
    echo.
    echo To start NOW: schtasks /run /tn "TradingBot"
    echo To stop:      schtasks /end /tn "TradingBot"
    echo To remove:    schtasks /delete /tn "TradingBot" /f
    echo.
    echo Log file: %LOG_FILE%
) else (
    echo.
    echo ERROR: Failed to create task. Try running as Administrator.
    echo.
    echo Manual alternative:
    echo   1. Open Task Scheduler
    echo   2. Create Basic Task
    echo   3. Name: "TradingBot"
    echo   4. Trigger: "At log on"
    echo   5. Action: "Start a program"
    echo   6. Program: pythonw.exe
    echo   7. Arguments: "%WATCHDOG%"
)

pause
