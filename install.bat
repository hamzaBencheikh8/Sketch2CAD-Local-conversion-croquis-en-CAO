@echo off
REM Installation de Sketch2CAD-Local dans un environnement virtuel local (.venv)
cd /d "%~dp0"
py -3.12 -m venv .venv 2>nul || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m sketch2cad doctor
echo.
echo Installation terminee. Pour les modeles locaux (optionnel), installez Ollama puis :
echo    ollama pull qwen2.5vl:3b
echo    ollama pull qwen2.5-coder:3b
echo.
echo Pour tester : double-cliquez web.bat (interface sur http://127.0.0.1:8765)
pause
