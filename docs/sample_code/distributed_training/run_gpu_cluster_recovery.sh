#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster_recovery.sh DATA_PATH"
echo "For example: bash run_gpu_cluster_recovery.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

export MS_ENABLE_RECOVERY=1      # Enable recovery
export MS_RECOVERY_PATH=/XXX/XXX # Set recovery path

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu_recovery.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
export MS_NODE_ID=sched               # The node id for Scheduler
pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > scheduler.log 2>&1 &

# Launch 8 workers.
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    export MS_NODE_ID=worker_$i           # The node id for Workers
    pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > worker_$i.log 2>&1 &
done
