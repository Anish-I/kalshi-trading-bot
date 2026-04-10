@echo off
REM Launches all kalshi bot processes hidden (no console windows)
REM using pythonw.exe. Weather trader intentionally excluded — re-enable
REM only after calibration fix ships.
setlocal
set PYW=C:\Users\ivatu\AppData\Local\Programs\Python\Python311\pythonw.exe
set REPO=C:\Users\ivatu\kalshi-trading-bot
cd /d "%REPO%"

start "" /B "%PYW%" "%REPO%\scripts\collect_data.py"
start "" /B "%PYW%" "%REPO%\scripts\crypto_combined_trader.py" --mode sim
start "" /B "%PYW%" "%REPO%\scripts\run_fed_trader.py" --mode paper
start "" /B "%PYW%" "%REPO%\scripts\archive_kalshi_markets.py"
start "" /B "%PYW%" "%REPO%\scripts\start_dashboard.py"

echo Launched 5 hidden background processes via pythonw.exe
echo (weather_trade.py excluded — calibration fix pending)
endlocal
