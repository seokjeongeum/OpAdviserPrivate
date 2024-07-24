export PYTHONPATH=$PWD
cd scripts || exit
python optimize.py --config=config_sysbenchrw.ini
python optimize.py --config=config_sysbenchwo.ini
python optimize.py --config=config_sysbenchro.ini
python optimize.py --config=config_twitter.ini
