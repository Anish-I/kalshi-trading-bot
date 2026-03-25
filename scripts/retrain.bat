@echo off
cd /d C:\Users\ivatu\kalshi-trading-bot
python scripts/scheduled_retrain.py >> D:\kalshi-data\logs\retrain.log 2>&1
