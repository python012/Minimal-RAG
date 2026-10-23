@echo off
REM Clear ChromaDB vector database
echo Clearing database...
python data_loader.py --clear
echo.
echo Done!
pause
