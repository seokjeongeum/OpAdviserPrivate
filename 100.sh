#!/bin/bash
chmod +x ./job.sh
./job.sh
update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
python -m pip install --upgrade pip
pip install --user --upgrade setuptools
pip install --upgrade wheel
python -m pip install -r requirements.txt
python -m pip install .
for optimize_method in "DDPG" "GA" "MBO" "SMAC"; do
  lowercase="${optimize_method,,}"
  for knob_num in 5 11; do

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
