#!/bin/bash
chmod +x ./tpch.sh
./tpch.sh
for optimize_method in "DDPG" "GA" "MBO" "SMAC"; do
  lowercase="${optimize_method,,}"
  for knob_num in 50 139; do
    python3 scripts/optimize.py \
    --config=scripts/cluster.ini \
    --knob_config_file=scripts/experiment/gen_knobs/OLTP.json \
    --knob_num=$knob_num \
    --dbname=tpch \
    --workload=tpch \
    --oltpbench_config_xml=~/jeseok/oltpbench/config/sample_tpcc_config.xml \
    --task_id="tpch_${lowercase}_${knob_num}" \
    --optimize_method="$optimize_method"
  done
done
