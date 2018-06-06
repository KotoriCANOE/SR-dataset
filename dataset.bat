cd /d "%~dp0"

python save.py "K:\Dataset.SR\Train" "K:\Dataset.SR\TrainPP" --training --num-epochs 32 --random-seed 0
python save.py "K:\Dataset.SR\Test" "K:\Dataset.SR\TestPP" --testing --num-epochs 1 --random-seed 0

pause
