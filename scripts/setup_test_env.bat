@echo off
REM Set up a Python virtual environment and install dependencies for Culture.ai tests

set VENV_DIR=.venv

if not exist %VENV_DIR% (
    REM Guarantee Python 3.10 is used for the virtual environment
    py -3.10 -m venv %VENV_DIR%
)

call %VENV_DIR%\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

python - <<PY
import importlib.util, subprocess, sys
if importlib.util.find_spec("xdist") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest-xdist"])
PY

echo Environment ready. Run pytest with %VENV_DIR%\Scripts\python -m pytest
