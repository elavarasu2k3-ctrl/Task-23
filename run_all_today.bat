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
python -c "import tensorflow, grpc, tensorflow_serving, pandas, matplotlib, nbconvert, tensorboard" >nul 2>&1
if errorlevel 1 (
  echo Installing packages, this takes a few minutes...
  pip install tensorflow tensorflow-serving-api grpcio requests pandas matplotlib notebook nbconvert tensorboard >> run_log.txt 2>&1
)
echo Running Day 1 notebook (10-20 minutes). Keep this window open...
jupyter nbconvert --to notebook --execute --inplace --allow-errors --ExecutePreprocessor.timeout=1800 Task23_Day1_Phases1-8.ipynb >> run_log.txt 2>&1
echo Day1 finished: %date% %time% >> run_log.txt
echo Running Day 2 capstone notebook (10-20 minutes)...
jupyter nbconvert --to notebook --execute --inplace --allow-errors --ExecutePreprocessor.timeout=1800 Task23_Day2_Capstone.ipynb >> run_log.txt 2>&1
echo Finished: %date% %time% >> run_log.txt
echo.
echo DONE. Outputs saved into both notebooks  (log: run_log.txt)
pause
