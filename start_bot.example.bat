@echo off
REM ============================================================
REM  MODELO — copie este arquivo para start_bot.bat e preencha
REM  com o token real. start_bot.bat NAO vai pro GitHub (esta no
REM  .gitignore), entao o token fica so na sua maquina.
REM ============================================================
cd /d "%~dp0"

set TELEGRAM_TOKEN=COLE_SEU_TOKEN_AQUI
set TELEGRAM_CHAT_ID=COLE_SEU_CHAT_ID_AQUI

REM puxa o modelo mais recente (retreinado toda segunda no GitHub Actions)
git pull

python alert_ml.py

pause
