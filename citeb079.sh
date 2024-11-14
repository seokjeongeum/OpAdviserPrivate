#!/bin/bash
chmod +x ./cluster.sh
./cluster.sh
mysql -ppassword -e"drop database tpcc;"
mysql -ppassword -e"create database tpcc;"
~/jeseok/oltpbench/oltpbenchmark -b tpcc -c ~/jeseok/oltpbench/config/sample_tpcc_config.xml  --create=true --load=true
for optimize_method in "DDPG" "GA" "MBO" "SMAC"; do
  lowercase="${optimize_method,,}"
  for knob_num in 7 138 10; do
    python3 scripts/optimize.py \
    --config=scripts/cluster.ini \
    --knob_config_file=scripts/experiment/gen_knobs/moreworkloads/tpcc_shap.json \
    --knob_num=$knob_num \
    --dbname=tpcc \
    --workload=oltpbench_tpcc \
    --oltpbench_config_xml=~/jeseok/oltpbench/config/sample_tpcc_config.xml \
    --task_id="tpcc_${lowercase}_${knob_num}" \
    --optimize_method="$optimize_method"
  done
done
