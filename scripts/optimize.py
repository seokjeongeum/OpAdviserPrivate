from autotune.utils.config import parse_args
from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.tuner import DBTuner
import argparse

if __name__ == '__main__':
    #2024-11-22 softmax weight
    for b in [True,False]:
    #2024-11-22 softmax weight
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='scripts/config.ini', help='config file')
        # 2024-11-19 code for clusters
        # parser.add_argument('--knob_config_file', type=str,default='scripts/experiment/gen_knobs/SYSBENCH_randomforest.json')
        # parser.add_argument('--knob_num', type=int,default=118)
        # parser.add_argument('--dbname', type=str,default='sbread')
        # parser.add_argument('--workload', type=str,default='sysbench')
        # parser.add_argument('--oltpbench_config_xml', type=str,default='/oltpbench/config/sample_twitter_config.xml')
        parser.add_argument('--task_id', type=str, default=f"oltpbench_twitter_smac_{b}")
        # parser.add_argument('--optimize_method', type=str, default='DDPG')
        #2024-11-22 softmax weight
        parser.add_argument('--softmax_weight', type=bool, default=b)
        #2024-11-22 softmax weight
        # 2024-11-19 code for clusters
        opt = parser.parse_args()


        args_db, args_tune = parse_args(opt.config)
        # 2024-11-19 code for clusters
        # args_db, args_tune = parse_args(opt.config,opt.knob_config_file)
        # args_db['knob_config_file']=opt.knob_config_file
        # args_db['knob_num']=opt.knob_num
        # args_db['dbname'] = opt.dbname
        # args_db['workload'] = opt.workload
        # args_db['oltpbench_config_xml'] = opt.oltpbench_config_xml
        args_tune['task_id'] = opt.task_id
        # args_tune['optimize_method'] = opt.optimize_method
        # args_tune['initial_tunable_knob_num']=opt.knob_num
        #2024-11-22 softmax weight
        args_tune['softmax_weight']=opt.softmax_weight
        #2024-11-22 softmax weight
        # 2024-11-19 code for clusters
        if args_db['db'] == 'mysql':
            db = MysqlDB(args_db)
        elif args_db['db'] == 'postgresql':
            db = PostgresqlDB(args_db)

        env = DBEnv(args_db, args_tune, db)
        tuner = DBTuner(args_db, args_tune, env)
        tuner.tune()
    #code for error case analysis
        # tuner.f()
    #code for error case analysis

