#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_inference.sh"
echo "=============================================================================================================="

msrun --worker_num=8 \
      --local_worker_num=8 \
      --master_addr=127.0.0.1 \
      --master_port=10969 \
      --join=True \
      --log_dir=./pipeline_inference_logs \
      python "distributed_pipeline_parallel_inference.py"
