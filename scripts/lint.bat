@echo off
echo Running Black...
black src\ tests\

echo Running isort...
isort src\ tests\

echo Running Flake8...
flake8 src\ tests\

echo Running Mypy...
mypy src\ tests\

echo Linting and formatting complete.
pause 