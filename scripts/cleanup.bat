@echo off
echo Cleaning up temporary directories...

:: Check if temp directory exists, if not create it
if not exist temp mkdir temp

:: Move test directories to temp folder
echo Moving temp test directories...
if exist test_memory_utility_score_* move test_memory_utility_score_* temp\
if exist test_mus_pruning_* move test_mus_pruning_* temp\
if exist temp_extract move temp_extract temp\
if exist __pycache__ move __pycache__ temp\

:: Clean Python cache files
echo Cleaning Python cache files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" echo Removing: %%d && rd /s /q "%%d"

echo Cleanup complete!
pause 