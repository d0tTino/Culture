@echo off
REM Aggressive cleanup for Python/AI project
rmdir /s /q __pycache__
rmdir /s /q .mypy_cache
rmdir /s /q .ruff_cache
rmdir /s /q .pytest_cache
rmdir /s /q htmlcov
rmdir /s /q logs
rmdir /s /q temp
rmdir /s /q chroma_db
rmdir /s /q scripts\temp
rmdir /s /q data\logs
rmdir /s /q archives\__pycache__
rmdir /s /q scripts\archive\__pycache__
rmdir /s /q src\agents\__pycache__
rmdir /s /q src\agents\core\__pycache__
rmdir /s /q src\agents\memory\__pycache__
rmdir /s /q src\agents\dspy_programs\__pycache__
rmdir /s /q tests\__pycache__
rmdir /s /q tests\unit\__pycache__
rmdir /s /q tests\integration\__pycache__
del /s /q *.log
 del /s /q *.txt
 del /s /q *.out
 del /s /q *.pyc
 del /s /q *.pyo
 del /s /q *.pyd
 del /s /q *.pdb
 del /s /q .coverage*
 del /s /q final_test_suite_output.txt
 del /s /q pytest_*.txt
 del /s /q coverage_*.txt
 del /s /q archives\*.zip

echo Cleanup complete!
pause 