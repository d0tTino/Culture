@echo off
REM Run the walking vertical slice demo. Activates venv or .venv if present.
REM Load key-value pairs from .env if the file exists
REM Lines beginning with '#' are ignored
if exist .env (
    for /f "tokens=1,* delims== eol=#" %%A in (".env") do (
        set "%%A=%%B"
    )
)
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
python -m examples.walking_vertical_slice

