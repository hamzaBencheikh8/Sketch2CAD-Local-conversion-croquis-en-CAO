@echo off
REM Lance l'interface web de test sur http://127.0.0.1:8765 (Ctrl+C pour arreter)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m sketch2cad web
pause
