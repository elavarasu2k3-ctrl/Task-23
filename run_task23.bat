@echo off
REM Runs every cell of the Task 23 notebook and saves the outputs into the same file.
cd /d "%~dp0"
echo ==== Task 23 runner ==== > run_log.txt
docker info >nul 2>&1
if errorlevel 1 (
  echo Docker is not running. Start Docker Desktop, wait for "Engine running", then run this again.
  echo Docker not running >> run_log.txt
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
python -c "import tensorflow, grpc, tensorflow_serving, pandas, matplotlib, nbconvert" >nul 2>&1
if errorlevel 1 (
  echo Installing packages, this takes a few minutes...
  pip install tensorflow tensorflow-serving-api grpcio requests pandas matplotlib notebook nbconvert >> run_log.txt 2>&1
)
echo Running all cells (10-20 minutes). Keep this window open...
jupyter nbconvert --to notebook --execute --inplace --allow-errors --ExecutePreprocessor.timeout=1800 Task23_Day1_Phases1-8.ipynb >> run_log.txt 2>&1
echo Finished: %date% %time% >> run_log.txt
echo.
echo DONE. Outputs saved into Task23_Day1_Phases1-8.ipynb  (log: run_log.txt)
pause
