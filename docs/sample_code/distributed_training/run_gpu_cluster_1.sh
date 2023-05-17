#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 1-4 workers.
for((i=0;i<4;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &
done

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
pytest -s -v ./resnet50_distributed_training_gpu.py > scheduler.log 2>&1 &