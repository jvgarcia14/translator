@echo off
setlocal
cd /d "%~dp0"
py -3.12 --version >nul 2>&1 || (
  echo Python 3.12 is required.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Installation completed successfully.
pause
exit /b 0
:error
echo.
echo Installation failed. Review the error above.
pause
exit /b 1
