@echo off
set FORMAT=0
if "%1"=="--format" set FORMAT=1
if "%1"=="-f" set FORMAT=1

if %FORMAT%==1 (
    echo Running Ruff format...
    ruff format src\ tests\
)

echo Running Ruff check...
ruff check src\ tests\

echo Running Black...
black src\ tests\

echo Running Mypy...
mypy src\ tests\

echo Linting and formatting complete.
