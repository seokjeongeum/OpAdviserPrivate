#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast data collection script for CPU/IO resource prediction model training.
Collects 60-80 samples with resource monitoring enabled.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from autotune.utils.config import parse_args
from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.utils.history_container import HistoryContainer
from autotune.utils.config_space import ConfigurationSpace
from autotune.knobs import initialize_knobs
from autotune.utils.config_space.util import convert_configurations_to_array
from autotune.tuner import DBTuner


def create_collection_config(output_dir='resource_data'):
    """Create a config file for fast data collection."""
    config_content = """[database]
db = mysql
host = localhost
port = 3306
user = root
passwd = password
sock = /var/run/mysqld/mysqld.sock
cnf = scripts/template/experiment_normandy.cnf
mysqld = /usr/sbin/mysqld
pg_ctl = /usr/bin/pg_ctl
pgdata = /var/lib/postgresql/data
postgres = /usr/bin/postgres

# CPU/IO focused: Use dynamic knobs only
knob_config_file = scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json
knob_num = 15

# Workload settings
dbname = sbrw
workload = sysbench
oltpbench_config_xml = 
workload_type = sbrw
thread_num = 40

# ULTRA-FAST: Minimal benchmark time
workload_warmup_time = 5
workload_time = 30

# Mode settings - CRITICAL: online_mode=True for speed
remote_mode = False
ssh_user = 
online_mode = True
isolation_mode = False
pid = 0
lhs_log = 
cpu_core = 

[tune]
task_id = resource_data_collection

performance_metric = ['tps']
reference_point = [None, None]
constraints = 

# Collection settings
max_runs = 60
optimize_method = SMAC
space_transfer = False
auto_optimizer = False

selector_type = shap
initial_runs = 60
initial_tunable_knob_num = 15
incremental = none
incremental_every = 10
incremental_num = 2

acq_optimizer_type = random
batch_size = 1
mean_var_file = 
params = 
tr_init = True
replay_memory = 

transfer_framework = none
data_repo = repo

only_knob = False
only_range = False
latent_dim = 0
"""
    
    config_path = os.path.join(output_dir, 'collection_config.ini')
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)
    return config_path


def collect_data(
    config_file: str,
    num_samples: int = 60,
    output_dir: str = 'resource_data',
    append: bool = True,
    workload_time_s: int = 30,
    workload_warmup_time_s: int = 5,
) -> list:
    """
    Collect resource data (CPU, ReadIO, WriteIO) for model training.
    
    Parameters:
    -----------
    config_file : str
        Path to configuration file
    num_samples : int
        Number of samples to collect (default: 60)
    output_dir : str
        Directory to save collected data
    append : bool
        If True, append to existing data; if False, overwrite (default: True)
    
    Returns:
    --------
    collected_data : list
        List of collected data points
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting data collection...")
    print(f"Target: {num_samples} samples")
    print(f"Output directory: {output_dir}")
    
    # Parse configuration
    args_db, args_tune = parse_args(config_file)
    
    # Override settings for collection.
    #
    # NOTE: Longer runs reduce metric noise and typically improve CPU MAPE,
    # but take more time per sample.
    args_db['workload_time'] = str(int(workload_time_s))
    args_db['workload_warmup_time'] = str(int(workload_warmup_time_s))
    args_db['online_mode'] = 'True'
    args_tune['max_runs'] = str(num_samples)
    args_tune['initial_runs'] = str(num_samples)  # Use random sampling
    
    # Initialize database and environment
    if args_db['db'] == 'mysql':
        db = MysqlDB(args_db)
    elif args_db['db'] == 'postgresql':
        db = PostgresqlDB(args_db)
    else:
        raise ValueError(f"Unsupported database type: {args_db['db']}")
    
    env = DBEnv(args_db, args_tune, db)
    
    # Create tuner to access setup_configuration_space method
    tuner = DBTuner(args_db, args_tune, env)
    config_space = tuner.setup_configuration_space(args_db['knob_config_file'], int(args_db['knob_num']))
    
    # Create history container
    history_container = HistoryContainer(
        task_id=args_tune['task_id'],
        config_space=config_space,
        num_constraints=len(env.constraints)
    )
    
    # Generate random configurations
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating {num_samples} random configurations...")
    configs = []
    for i in range(num_samples):
        config = config_space.sample_configuration()
        configs.append(config)
    
    # Collect data for each configuration
    collected_data = []
    start_time = time.time()
    
    for i, config in enumerate(configs):
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sample {i+1}/{num_samples}")
        print(f"Configuration: {config.get_dictionary()}")
        
        try:
            # Apply knobs
            env.apply_knobs(config.get_dictionary())
            
            # Get states with resource collection enabled
            benchmark_timeout, external_metrics, internal_metrics, resource = env.get_states(collect_resource=1)
            
            # Extract resource metrics
            cpu = resource[0]
            read_io = resource[1]
            write_io = resource[2]
            
            # Validate data
            if cpu <= 0 or read_io < 0 or write_io < 0:
                print(f"  Warning: Invalid resource metrics (cpu={cpu}, readIO={read_io}, writeIO={write_io}), skipping...")
                continue
            
            # Store data
            data_point = {
                'configuration': config.get_dictionary(),
                'external_metrics': external_metrics,
                'internal_metrics': list(internal_metrics),
                'resource': {
                    'cpu': cpu,
                    'readIO': read_io,
                    'writeIO': write_io,
                    'virtualMem': resource[3],
                    'physicalMem': resource[4],
                    'dirty': resource[5],
                    'hit': resource[6],
                    'data': resource[7]
                },
                'workload': {
                    'type': args_db['workload'],
                    'workload_type': args_db.get('workload_type', 'sbrw'),
                    'threads': int(args_db.get('thread_num', 40))
                },
                'trial_state': 'SUCCESS' if not benchmark_timeout else 'TIMEOUT'
            }
            
            collected_data.append(data_point)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (num_samples - i - 1)
            
            print(f"  CPU: {cpu:.2f}%, ReadIO: {read_io:.2f} MB/s, WriteIO: {write_io:.2f} MB/s")
            print(f"  Progress: {i+1}/{num_samples} | Elapsed: {elapsed/60:.1f}min | Remaining: ~{remaining/60:.1f}min")
            
            # Save intermediate results every 10 samples
            if (i + 1) % 10 == 0:
                save_data(collected_data, output_dir, intermediate=True, append=append)
        
        except Exception as e:
            print(f"  Error collecting sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final data
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Collection complete!")
    print(f"Collected {len(collected_data)} valid samples out of {num_samples} attempts")
    
    save_data(collected_data, output_dir, intermediate=False, append=append)
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per sample: {total_time/len(collected_data):.1f} seconds")
    
    return collected_data


def load_existing_data(output_dir: str) -> list:
    """
    Load existing data from output directory if available.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing existing data file
    
    Returns:
    --------
    existing_data : list
        List of existing data points, empty if no file exists
    """
    filename = os.path.join(output_dir, 'resource_data.json')
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                existing = json.load(f)
            existing_data = existing.get('data', [])
            print(f"  Loaded {len(existing_data)} existing samples from {filename}")
            return existing_data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not load existing data: {e}")
            return []
    return []


def save_data(data: list, output_dir: str, intermediate: bool = False, 
              append: bool = True) -> None:
    """
    Save collected data to JSON file.
    
    Parameters:
    -----------
    data : list
        New data points to save
    output_dir : str
        Output directory path
    intermediate : bool
        If True, save to intermediate file (default: False)
    append : bool
        If True, append to existing data; if False, overwrite (default: True)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if intermediate:
        filename = os.path.join(output_dir, 'resource_data_intermediate.json')
    else:
        filename = os.path.join(output_dir, 'resource_data.json')
    
    # Load and merge with existing data if append mode
    if append and not intermediate:
        existing_data = load_existing_data(output_dir)
        combined_data = existing_data + data
        print(f"  Accumulated: {len(existing_data)} existing + {len(data)} new = {len(combined_data)} total samples")
    else:
        combined_data = data
    
    output = {
        'info': {
            'num_samples': len(combined_data),
            'collection_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': ['cpu', 'readIO', 'writeIO']
        },
        'data': combined_data
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"  Saved {len(combined_data)} samples to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Collect resource data for model training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if not provided, creates default)')
    parser.add_argument('--num_samples', type=int, default=60,
                       help='Number of samples to collect (default: 60)')
    parser.add_argument('--output_dir', type=str, default='resource_data',
                       help='Output directory (default: resource_data)')
    parser.add_argument('--no-append', dest='append', action='store_false',
                       help='Overwrite existing data instead of appending (default: append)')
    parser.add_argument('--workload_time', type=int, default=30,
                       help='Benchmark run time in seconds per sample (default: 30). '
                            'Increase (e.g., 90-180) to reduce CPU measurement noise.')
    parser.add_argument('--workload_warmup_time', type=int, default=5,
                       help='Warmup time in seconds per sample (default: 5). '
                            'Increase (e.g., 10-30) for more stable CPU readings.')
    parser.set_defaults(append=True)
    
    args = parser.parse_args()
    
    # Show existing data info
    if args.append:
        existing = load_existing_data(args.output_dir)
        if existing:
            print(f"Append mode: Will add to {len(existing)} existing samples")
        else:
            print("Append mode: No existing data found, starting fresh")
    else:
        print("Overwrite mode: Will replace any existing data")
    
    # Create config if not provided
    if args.config is None:
        print("No config file provided, creating default config...")
        args.config = create_collection_config(args.output_dir)
    
    # Collect data
    collect_data(
        args.config,
        args.num_samples,
        args.output_dir,
        args.append,
        workload_time_s=args.workload_time,
        workload_warmup_time_s=args.workload_warmup_time,
    )


if __name__ == '__main__':
    main()

