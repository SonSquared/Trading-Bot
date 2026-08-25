@echo off
cd /d "C:\Users\SonSon\Desktop\Trading Bot"
"D:\Program Files\Python\python.exe" -u scripts\run_full_optimization.py --resume --checkpoint-every 500 > data\results\optimization_run.log 2>&1
echo Optimization finished with exit code %ERRORLEVEL% >> data\results\optimization_run.log
