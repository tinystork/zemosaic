@echo off
set PYTHON_EXEC=%~dp0\.venv\Scripts\python.exe
if exist "%PYTHON_EXEC%" (
  set PYTHON_CMD="%PYTHON_EXEC%"
) else (
  set PYTHON_CMD="python"
)
%PYTHON_CMD% detect_pyside.py
pause
