@echo off
rem Train in a visible console: resumes temp\train\%1 if it has a state.pt, else starts it fresh.
rem   trainer\train.cmd hoi4-v1
cd /d "%~dp0.."
set NAME=%1
if "%NAME%"=="" set NAME=hoi4-v1
set PY=%USERPROFILE%\.conda\envs\py313\python.exe
set PYTHONWARNINGS=ignore
title JevAI training %NAME%
if exist "temp\train\%NAME%\state.pt" (
  "%PY%" -m trainer.train --out temp/train/%NAME% --resume
) else (
  "%PY%" -m trainer.train --out temp/train/%NAME% --bf16 --device xpu --bs 16 --accum 2 --epochs 1 --eval-every 1000 --save-every 500 --val-states 1500
)
echo.
echo Training exited with code %ERRORLEVEL%. Log: temp\train\%NAME%\train.log
pause
