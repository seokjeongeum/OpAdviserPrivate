#!/bin/bash
chmod +x ./cluster.sh
chmod +x ./job.sh
./cluster.sh
./job.sh
for optimize_method in "DDPG" "GA" "MBO" "SMAC"; do
  lowercase="${optimize_method,,}"
  for knob_num in 74 84; do

    python3 scripts/optimize.py \
    --config=scripts/cluster.ini \
    --knob_config_file=scripts/experiment/gen_knobs/JOB_shap.json \
    --knob_num=$knob_num \
    --dbname=imdbload \
    --workload=job \
    --task_id="job_${lowercase}_${knob_num}" \
    --optimize_method="$optimize_method"
  done
done
chmod +x ./cluster2.sh
./cluster2.sh

