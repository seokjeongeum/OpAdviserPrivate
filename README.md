# OpAdviser: An Efficient Transfer Learning Based Configuration Adviser for Database Tuning

**OpAdviser** is a customized and efficient tuning system that  addresses the search space construction and the search optimizer selection  problems for database configuration tuning



## Installation 
Installation Requirements:
- Python >= 3.6 

 ```shell
   git clone git@github.com:Blairruc-pku/OpAdviser.git && cd OpAdviser
   pip install -r requirements.txt
   pip install .
   ```




## Preparation 
####  Workload Preparation 
Please reffer to the <a href="https://github.com/Blairruc-pku/OpAdviser/blob/main/documents/workload_prepare.md" target="_blank" rel="nofollow">details instuction</a>  for preparing the workloads.
####  Database Connection Setup
To provide the database connection information, the users need to edit the `config_auto.ini`.
```ini
db = mysql
host = 127.0.0.1
port = 3306
user = root
passwd =
  ```

## Quick Start

 
1. Specify the tuning objective in `config_auto.ini`. Here are some examples.


    Performance tuning, e.g., maximizing throughputs.
    ```ini
    task_id = performance1
    performance_metric = ['tps']
    ```
    
    Resource-oriented tuning, e.g., minimizing  cpu resource while throughputs > 200 txn/s and 95th percentile latency < 60 sec.

    ```ini
    task_id = resource1
    performance_metric = ['-cpu']
    #constraints: Non-positive constraint values (”<=0”) imply feasibility.
    constraints = ["200 - tps", "latency - 60"]
    ```

    Multiple objective tuning, e.g., maximizing throughput and minimizing I/O.
    ```ini
    task_id = mutiple1
    performance_metric = ['tps', '-cpu]
    reference_point = [0, 100]
   ```

2. Conduct Tuning.
    ```bash
    cd scripts
    python optimize.py  --config=config_performance.ini
    ```

    For more information, please refer to the <a href="https://github.com/Blairruc-pku/DBTuner/blob/main/documents/tuning_setting.md#specific-tuning-setting" target="_blank" rel="nofollow">specific tuning settings </a>. 
    
    
## Related Publications

**Facilitating Database Tuning with Hyper-Parameter Optimization: A Comprehensive Experimental Evaluation**[[PDF](https://arxiv.org/abs/2110.12654)]<br>
Xinyi Zhang, Zhuo Chang, Yang Li, Hong Wu, Jian Tan, Feifei Li, Bin Cui.<br>
The 48th International Conference on Very Large Data Bases .<br>
***VLDB 2022, CCF-A</font></b>***

**Towards Dynamic and Safe Configuration Tuning for Cloud Databases**[[PDF](https://arxiv.org/abs/2203.14473)]<br>
Xinyi Zhang, Hong Wu, Yang Li, Jian Tan, Feifei Li, Bin Cui.<br>
ACM Conference on Management of Data .<br>
***SIGMOD 2022, CCF-A</font></b>***

**ResTune: Resource Oriented Tuning Boosted by Meta-Learning
for Cloud Databases**[[PDF](https://dl.acm.org/doi/10.1145/3448016.3457291)]<br>
Xinyi Zhang, Hong Wu, Zhuo Chang, Shuowei Jin, Jian Tan, Feifei Li,
Tieying Zhang, and Bin Cui.<br>
ACM Conference on Management of Data .<br>
***SIGMOD 2021, CCF-A</font></b>***

## Contact

If you have any technical questions, please submit new issues.

If you have any other questions, please contact Xinyi Zhang[zhang_xinyi@pku.edu.cn] and Zhuo Chang[z.chang@pku.edu.cn].
## **License**

The entire codebase is under [MIT license](LICENSE).
