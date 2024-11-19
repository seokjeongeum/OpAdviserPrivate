#!/bin/bash
workload="ycsb"
cd /
rm -rf oltpbench && \
  git clone https://github.com/oltpbenchmark/oltpbench.git
cd /workspaces/OpAdviserPrivate
cp -r oltpbench_files/. /oltpbench
cd /oltpbench && \
    ant bootstrap && \
    ant resolve && \
    ant build && \
    chmod 777 /oltpbench/*
mysql -ppassword -e"drop database ${workload};"
mysql -ppassword -e"create database ${workload};"
/oltpbench/oltpbenchmark -b $workload -c /oltpbench/config/sample_${workload}_config.xml  --create=true --load=true
cd ~/OpAdviserPrivate
update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
python -m pip install --upgrade pip
pip install --user --upgrade setuptools
pip install --upgrade wheel
python -m pip install -r requirements.txt
python -m pip install .
for optimize_method in "DDPG" "GA" "MBO" "SMAC"; do
  lowercase="${optimize_method,,}"
  for knob_num in 174 84; do
    python3 scripts/optimize.py \
    --config=scripts/cluster.ini \
    --knob_config_file=scripts/experiment/gen_knobs/moreworkloads/${workload}_lhs_shap.json \
    --knob_num=$knob_num \
    --dbname=${workload} \
    --workload=oltpbench_${workload} \
    --oltpbench_config_xml=~/jeseok/oltpbench/config/sample_${workload}_config.xml \
    --task_id="${workload}_${lowercase}_${knob_num}" \
    --optimize_method="$optimize_method"
  done
done
