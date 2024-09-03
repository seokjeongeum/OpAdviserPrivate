export PYTHONPATH=$PWD
cd scripts || exit
python optimize.py --config=config_batchmc.ini
python optimize.py --config=config_cmaes.ini
python optimize.py --config=config_localrandom.ini
python optimize.py --config=config_mesmo.ini
python optimize.py --config=config_randomscipy.ini
python optimize.py --config=config_scipyglobal.ini
python optimize.py --config=config_stagedbatchscipy.ini
python optimize.py --config=config_usemo.ini
