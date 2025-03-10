import json
import os
import sys
from collections import  defaultdict

import numpy as np
from autotune.transfer.tlbo.rgpe import RGPE
from autotune.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter
from autotune.utils.config_space.util import convert_configurations_to_array
from autotune.utils.util_funcs import check_random_state
from autotune.workload_map import WorkloadMapping
from autotune.pipleline.pipleline import PipleLine
from .knobs import ts, logger, initialize_knobs
from .utils.parser import  get_hist_json
from autotune.utils.history_container import HistoryContainer, load_history_from_filelist
import pdb
from ConfigSpace.hyperparameters import NormalFloatHyperparameter
from autotune.utils.config_space import Configuration
class DBTuner:
    def __init__(self, args_db, args_tune, env):
        self.env = env
        self.create_output_folders()
        self.args_tune = args_tune
        self.args_db = args_db
        self.method = args_tune['optimize_method']
        self.y_variable = env.y_variable
        self.transfer_framework = args_tune['transfer_framework']
        self.hc_path = self.args_tune['data_repo']
        self.space_transfer = eval(self.args_tune['space_transfer'])
        self.auto_optimizer = eval(self.args_tune['auto_optimizer'])

        self.hcL = list()
        self.history_workload_data = list()
        self.model_params_path = ''
        self.surrogate_type = None
        self.objs = eval(args_tune['performance_metric'])
        if args_tune['constraints'] is None or args_tune['constraints'] == '':
            self.constraints = []
        else:
            self.constraints = eval(args_tune['constraints'])

        # self.latent_dim=int(args_tune['latent_dim'])

        self.config_space = self.setup_configuration_space(args_db['knob_config_file'], int(args_db['knob_num']))

        self.acq_optimizer_type=self.args_tune['acq_optimizer_type']

        self.setup_transfer()


    def setup_configuration_space(self, knob_config_file, knob_num):
        # if self.latent_dim!=0:
        #     config_space = ConfigurationSpace()
        #     l=[]
        #     for i in range(self.latent_dim):
        #         l.append(NormalFloatHyperparameter(f'dim{i}',mu=0,sigma=1,lower=-1,upper=1))
        #     config_space.add_hyperparameters(l)
        #     return config_space

        KNOBS = initialize_knobs(knob_config_file, knob_num)
        knobs_list = []
        config_space = ConfigurationSpace()

        for name in KNOBS.keys():
            value = KNOBS[name]
            knob_type = value['type']
            if knob_type == 'enum':
                knob = CategoricalHyperparameter(name, [str(i) for i in value["enum_values"]], default_value= str(value['default']))
            elif knob_type == 'integer':
                min_val, max_val = value['min'], value['max']
                if KNOBS[name]['max'] > sys.maxsize:
                    knob = UniformIntegerHyperparameter(name, int(min_val / 1000), int(max_val / 1000),
                                                        default_value=int(value['default'] / 1000))
                else:
                    knob = UniformIntegerHyperparameter(name, min_val, max_val, default_value=value['default'])
            elif knob_type == 'float' or knob_type == 'real':
                min_val, max_val = value['min'], value['max']
                knob = UniformFloatHyperparameter(name, min_val, max_val, default_value=value['default'])
            else:
                raise ValueError('Invalid knob type!')

            knobs_list.append(knob)

        config_space.add_hyperparameters(knobs_list)

        return config_space

    def load_history(self, knob_num):
        files = os.listdir(self.hc_path)
        config_space = self.setup_configuration_space(self.args_db['knob_config_file'], int(self.args_db['knob_num']))
        for f in files:
            try:
                task_id = f.split('.')[0]
                fn = os.path.join(self.hc_path, f)
                history_container = HistoryContainer(task_id, config_space=config_space)
                history_container.load_history_from_json(fn)
                self.hcL.append(history_container)
            except:
                logger.info('load history failed for {}'.format(f))

    def setup_transfer(self):
        if self.transfer_framework == 'none':
            if self.method == 'SMAC':
                self.surrogate_type = 'prf'
            elif self.method == 'MBO':
                self.surrogate_type = 'gp'
            elif self.method == 'auto':
                self.surrogate_type = 'auto'
        elif self.transfer_framework in ['workload_map', 'rgpe']:
            self.load_history(int(self.args_db['knob_num']))
            if self.method == 'SMAC':
                method = 'prf'
            elif self.method == 'MBO':
                method = 'gp'
            else:
                raise ValueError('Invalid method for %s!' % self.transfer_framework)

            if self.transfer_framework == 'rgpe':
                self.surrogate_type = 'tlbo_rgpe_' + method
            else:
                self.surrogate_type = 'tlbo_mapping_' + method

        elif self.transfer_framework == 'finetune':
            if self.method != 'DDPG':
                raise ValueError('Invalid method for finetune!')

            self.model_params_path = self.args_tune['params']

        elif   self.transfer_framework == 'context':
            self.surrogate_type = 'context_prf'
            if self.method != 'SMAC':
                raise ValueError('We currently only support SMAC. Invalid method for context!')

        else:
            raise ValueError('Invalid string %s for transfer framework!' % self.transfer_framework)

        if len(self.hcL)==0 and self.space_transfer:
        # 2024-11-20: ground truth
        # if True:
        # 2024-11-20: ground truth
            self.load_history(-1)

        if self.auto_optimizer and self.args_tune['auto_optimizer_type'] == 'learned':
            self.history_workload_data = self.load_workload_data()

    def load_workload_data(self):
        file_dict = defaultdict(list)
        history_workload_data = list()
        workloadL= [ 'sysbench', 'twitter', 'job', 'tpch']
        workloadL.remove(self.args_db['workload'] if '_' not in self.args_db['workload'] else self.args_db['workload'].split('_')[1] )
        files = os.listdir(self.hc_path)
        config_space = self.setup_configuration_space(self.args_db['knob_config_file'], int(self.args_db['knob_num']))
        for f in files:
                task_id = f.split('.')[0]
                for workload in workloadL:
                    if workload in task_id:
                        break
                if not workload in task_id:
                    continue
                fn = os.path.join(self.hc_path, f)
                file_dict[workload].append(fn)

        for workload in file_dict.keys():
            history_container = load_history_from_filelist(workload, file_dict[workload], config_space)
            history_workload_data.append(history_container)

        return history_workload_data

    def tune(self):
        bo = PipleLine(self.env.step,
                       self.config_space,
                       num_objs=len(self.objs),
                       num_constraints=len(self.constraints),
                       optimizer_type=self.method,
                       max_runs=int(self.args_tune['max_runs']),
                       surrogate_type=self.surrogate_type,
                       history_bo_data=self.hcL,
                       acq_optimizer_type=self.acq_optimizer_type,  # 'random_scipy',#
                       selector_type=self.args_tune['selector_type'],
                       initial_runs=int(self.args_tune['initial_runs']),
                       incremental=self.args_tune['incremental'],
                       incremental_every=int(self.args_tune['incremental_every']),
                       incremental_num=int(self.args_tune['incremental_num']),
                       init_strategy='random_explore_first',
                       ref_point= self.env.reference_point,
                       task_id=self.args_tune['task_id'],
                       time_limit_per_trial=60 * 200,
                       num_hps_init=int(self.args_tune['initial_tunable_knob_num']),
                       num_metrics=self.env.db.num_metrics,
                       mean_var_file=self.args_tune['mean_var_file'],
                       batch_size=int(self.args_tune['batch_size']),
                       params=self.model_params_path,
                       space_transfer=self.space_transfer,
                       knob_config_file=self.args_db['knob_config_file'],
                       auto_optimizer=self.auto_optimizer,
                       auto_optimizer_type= self.args_tune['auto_optimizer_type'],
                       hold_out_workload=self.args_db['workload'],
                       history_workload_data=self.history_workload_data,
                       only_knob=eval(self.args_tune['only_knob']),
                       only_range=eval(self.args_tune['only_range']),
                    #    latent_dim=int(self.args_tune['latent_dim'])
                 #2024-11-22 softmax weight
                 softmax_weight=self.args_tune['softmax_weight'],
                 #2024-11-22 softmax weight
                       )
        
#         id=self.args_tune['task_id']
#         s=''
#         s+=f's = "{id}"'
#         s+='''
# '''
#         bo.rgpe = RGPE(bo.config_space, bo.history_bo_data, check_random_state(100).randint(2 ** 31 - 1), num_src_hpo_trial=-1, only_source=False)
#         similarity_list =[1 - i for i in bo.rgpe.get_ranking_loss(bo.history_container)]
#         s+=(f'similarity_list = {similarity_list}')
#         s+=('''
# ''')
#         with open(f"repo/history_{self.args_tune['task_id']}_ground_truth.json") as f2:
#             j = json.load(f2)["data"]
#             c = sorted(j, key=lambda x: x["external_metrics"].get("tps", 0))[-1]
#         c = c["configuration"]
#         c=Configuration(self.config_space, c)
#         c=convert_configurations_to_array([c])
#         Y=[x.get_transformed_perfs() for x in self.hcL+[bo.history_container]]
#         ranks=[]
#         for i,surrogate in enumerate(bo.rgpe.source_surrogates+[bo.rgpe.target_surrogate]):
#             mu,var=surrogate.predict(c)         
#             number=np.random.normal(mu, var)
#             temp_list=np.concatenate((Y[i],number.reshape(-1))) 
#             sorted_list=sorted(temp_list)
#             rank=sorted_list.index(number.item())
#             ranks.append( (len(temp_list)-rank)/len(temp_list))
#         s+=(f'ranks = {ranks}')    
#         s+=('''
# plt.scatter(similarity_list, ranks)
# plt.title(s)
# plt.show()
# ''')
#         print(s)

        history = bo.run()
#2024-11-11: code for experiment
        # history,history2 = bo.run()
#2024-11-11: code for experiment
        if history.num_objs == 1:
            import matplotlib.pyplot as plt
            history.plot_convergence()
            plt.savefig('%s.png' % history.task_id)
#2024-11-11: code for experiment
        # if history2.num_objs == 1:
        #     import matplotlib.pyplot as plt
        #     history2.plot_convergence()
        #     plt.savefig('%s.png' % history2.task_id)
#2024-11-11: code for experiment


    @staticmethod
    def create_output_folders():
        output_folders = ['log', ]
        for folder in output_folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

#code for error case analysis
    # def f(self):
    #     for task_id in [
    #         'oltpbench_tatp_smac',
    #         'oltpbench_tpcc_smac',
    #         # 'oltpbench_wikipedia_smac',
    #         'oltpbench_ycsb_smac',
    #         'sbread_smac',
    #         'sbrw_smac',
    #         'sbwrite_smac',
    #         'twitter_smac',
    #         ]:
    #         bo = PipleLine(self.env.step,
    #                     self.config_space,
    #                     num_objs=len(self.objs),
    #                     num_constraints=len(self.constraints),
    #                     optimizer_type=self.method,
    #                     max_runs=int(self.args_tune['max_runs']),
    #                     surrogate_type=self.surrogate_type,
    #                     history_bo_data=self.hcL,
    #                     acq_optimizer_type=self.acq_optimizer_type,  # 'random_scipy',#
    #                     selector_type=self.args_tune['selector_type'],
    #                     initial_runs=int(self.args_tune['initial_runs']),
    #                     incremental=self.args_tune['incremental'],
    #                     incremental_every=int(self.args_tune['incremental_every']),
    #                     incremental_num=int(self.args_tune['incremental_num']),
    #                     init_strategy='random_explore_first',
    #                     ref_point= self.env.reference_point,
    #                     task_id=task_id,
    #                     time_limit_per_trial=60 * 200,
    #                     num_hps_init=int(self.args_tune['initial_tunable_knob_num']),
    #                     num_metrics=self.env.db.num_metrics,
    #                     mean_var_file=self.args_tune['mean_var_file'],
    #                     batch_size=int(self.args_tune['batch_size']),
    #                     params=self.model_params_path,
    #                     space_transfer=self.space_transfer,
    #                     knob_config_file=self.args_db['knob_config_file'],
    #                     auto_optimizer=self.auto_optimizer,
    #                     auto_optimizer_type= self.args_tune['auto_optimizer_type'],
    #                     hold_out_workload=self.args_db['workload'],
    #                     history_workload_data=self.history_workload_data,
    #                     only_knob=eval(self.args_tune['only_knob']),
    #                     only_range=eval(self.args_tune['only_range']),
    #                     #    latent_dim=int(self.args_tune['latent_dim'])
    #                     )
    #         bo.get_compact_space()
#code for error case analysis
