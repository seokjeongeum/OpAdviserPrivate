export PYTHONPATH=$PWD
cd scripts || exit
#python optimize.py --config=config_localrandom.ini
python optimize.py --config=config_stagedbatchscipy.ini
