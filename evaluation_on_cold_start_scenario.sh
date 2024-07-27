export PYTHONPATH=$PWD
cd scripts || exit
python optimize.py --config=config_tpcc.ini
rm OpAdviser_history/*
python optimize.py --config=config_ycsb.ini
rm OpAdviser_history/*
python optimize.py --config=config_sysbenchro.ini
rm OpAdviser_history/*
python optimize.py --config=config_wikipedia.ini
rm OpAdviser_history/*
python optimize.py --config=config_sysbenchwo.ini
rm OpAdviser_history/*
python optimize.py --config=config_tatp.ini
rm OpAdviser_history/*
python optimize.py --config=config_sysbenchrw.ini
rm OpAdviser_history/*
python optimize.py --config=config_twitter.ini
rm OpAdviser_history/*
python optimize.py --config=config_voter.ini
rm OpAdviser_history/*
