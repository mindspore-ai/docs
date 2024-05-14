#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh"
echo "=============================================================================================================="

EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf output
mkdir output

# run Scheduler process
export MS_SERVER_NUM=8
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_SCHED
python train.py > output/scheduler.log 2>&1 &

# run Server processes
export MS_SERVER_NUM=8
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_PSERVER
for((server_id=0;server_id<${MS_SERVER_NUM};server_id++))
do
    python train.py > output/server_${server_id}.log 2>&1 &
done

# run Wroker processes
export MS_SERVER_NUM=8
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_WORKER
for((worker_id=0;worker_id<${MS_WORKER_NUM};worker_id++))
do
    python train.py > output/worker_${worker_id}.log 2>&1 &
done
