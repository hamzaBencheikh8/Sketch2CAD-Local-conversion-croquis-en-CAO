@echo off
REM Usage : run.bat chemin\vers\croquis.jpg [profil]
REM   profil = offline (defaut) | cpu | gpu4 | gpu8
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PROFILE=%2
if "%PROFILE%"=="" set PROFILE=offline
python -m sketch2cad --profile %PROFILE% run "%~1" -o "sortie\%~n1"
echo.
echo Fichiers ecrits dans sortie\%~n1
pause
