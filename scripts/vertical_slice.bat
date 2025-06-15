@echo off
REM Run the walking vertical slice demo. Activates venv if present.
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
python -m examples.walking_vertical_slice

