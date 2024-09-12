#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh"
echo "=============================================================================================================="

msrun --worker_num=8 --local_worker_num=8 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 high_dimension_tensor_parallel.py