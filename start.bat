@echo off
title Demucs Audio Stem Separator
cd /d "%~dp0"
echo ====================================================
echo   Demucs Audio Stem Separator Ishga Tushirilmoqda...
echo ====================================================
call .venv\Scripts\activate.bat
python run.py
pause

