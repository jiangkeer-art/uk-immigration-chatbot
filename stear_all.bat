@echo off
chcp 65001 >nul
cd /d "%~dp0"
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

start "Chroma DB1" cmd /k ".venv\Scripts\chroma.exe run --path ./immigration_db --host 127.0.0.1 --port 8000"

start "Chroma DB2" cmd /k ".venv\Scripts\chroma.exe run --path ./immigration_db2 --host 127.0.0.1 --port 8001"

timeout /t 5 /nobreak >nul

start "Streamlit" cmd /k ".venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501"

start "Monitor" cmd /k ".venv\Scripts\python.exe test1.py"

start "ngrok" cmd /k "ngrok http 8501"