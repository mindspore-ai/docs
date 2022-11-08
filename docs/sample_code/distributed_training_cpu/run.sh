#!/bin/bash
# run data parallel training on CPU

echo "=============================================================================================================="
echo "Please run the script with dataset path, such as: "
echo "bash run.sh DATA_PATH"
echo "For example: bash run.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8117

# Launch 1 scheduler.
export MS_ROLE=MS_SCHED
python3 resnet50_distributed_training.py >scheduler.txt 2>&1 &
echo "scheduler start success!"

# Launch 8 workers.
export MS_ROLE=MS_WORKER
for((i=0;i<${MS_WORKER_NUM};i++));
do
    python3 resnet50_distributed_training.py >worker_$i.txt 2>&1 &
    echo "worker ${i} start success with pid ${!}"
done
