import argparse
import pathlib

from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.tuner import DBTuner
from autotune.utils.config import parse_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_performance.ini', help='config file')
    opt = parser.parse_args()

    args_db, args_tune = parse_args(opt.config)

    for t in [
        ('sbtest', 'sysbench', '', 'readwrite'),
        ('sbread', 'sysbench', '', 'read'),
        ('sbwrite', 'sysbench', '', 'write'),
        ('imdbload', 'job', '', ''),
        ('twitter', 'oltpbench_twitter', '../oltpbench/config/sample_twitter_config.xml', ''),
        ('ycsb', 'oltpbench_ycsb', '../oltpbench/config/sample_ycsb_config.xml', ''),
        ('tpcc', 'oltpbench_tpcc', '../oltpbench/config/sample_tpcc_config.xml', ''),
        ('voter', 'oltpbench_voter', '../oltpbench/config/sample_voter_config.xml', ''),
        ('tpch', 'tpch', '', ''),
    ]:
        args_db['dbname'] = t[0]
        args_db['workload'] = t[1]
        args_db['oltpbench_config_xml'] = t[2]
        args_db['workload_type'] = t[3]
        knob_config_files = ['experiment/gen_knobs/mysql_all_197_32G.json']
        if args_db['workload'].startswith('oltpbench_'):
            if args_db['dbname'].startswith('ycsb'):
                knob_config_files.append('experiment/gen_knobs/moreworkloads/ycsb_lhs_shap.json')
            else:
                knob_config_files.append(f'experiment/gen_knobs/moreworkloads/{args_db["dbname"]}_shap.json')
        elif args_db['workload'] != 'tpch':
            knob_config_files.append(f'experiment/gen_knobs/{args_db["workload"]}_shap.json')
        for i in range(len(knob_config_files)):
            args_db['knob_config_file'] = knob_config_files[i]
            for knob_num in range(197, 3 - 1, -1):
                args_db['knob_num'] = knob_num
                args_tune[
                    'task_id'
                ] = f'{args_db["workload"]}_{"smac"}_{knob_num}_{pathlib.Path(args_db["knob_config_file"]).stem}'
                if args_db['db'] == 'mysql':
                    db = MysqlDB(args_db)
                elif args_db['db'] == 'postgresql':
                    db = PostgresqlDB(args_db)

                env = DBEnv(args_db, args_tune, db)
                tuner = DBTuner(args_db, args_tune, env)
                tuner.tune()
