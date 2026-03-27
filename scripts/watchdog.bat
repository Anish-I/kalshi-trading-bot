@echo off
REM Watchdog: checks if data collector is running, restarts if dead.
REM Run via Windows Task Scheduler every 5 minutes.

cd /d C:\Users\ivatu\kalshi-trading-bot

REM Check if collect_data.py is running
tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>NUL | findstr /I "python" >NUL
if errorlevel 1 (
    echo %date% %time% Collector not running. Starting... >> D:\kalshi-data\logs\watchdog.log
    start /B python scripts\collect_data.py >> D:\kalshi-data\logs\collector_watchdog.log 2>&1
) else (
    REM Check if collect_data specifically is running (not just any python)
    wmic process where "name='python.exe' and commandline like '%%collect_data%%'" get processid 2>NUL | findstr /R "[0-9]" >NUL
    if errorlevel 1 (
        echo %date% %time% Collector not found. Starting... >> D:\kalshi-data\logs\watchdog.log
        start /B python scripts\collect_data.py >> D:\kalshi-data\logs\collector_watchdog.log 2>&1
    )
)
