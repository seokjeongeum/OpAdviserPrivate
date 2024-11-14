#!/bin/bash
chmod +x ./cluster.sh
./cluster.sh
mysql -ppassword -e"drop database sbwrite;"
mysql -ppassword -e"create database sbwrite;"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=80  \
    --mysql-db=sbwrite  \
    oltp_write_only  \
    prepare
for optimize_method in "DDPG" "GA" "MBO" "SMAC"; do
  lowercase="${optimize_method,,}"
  for knob_num in 11 133 7; do
    python3 scripts/optimize.py \
    --config=scripts/cluster.ini \
    --knob_config_file=scripts/experiment/gen_knobs/SYSBENCH_randomforest.json \
    --knob_num=$knob_num \
    --dbname=sbwrite \
    --workload=sysbench \
    --task_id="sysbench_${lowercase}_${knob_num}" \
    --optimize_method="$optimize_method"
  done
done
