#!/bin/bash

DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
RANK_SIZE=$2

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    export MINDSPORE_HCCL_CONFIG_PATH=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    export MINDSPORE_HCCL_CONFIG_PATH=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./resnet50_distributed_training.py ./resnet.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v ./resnet50_distributed_training.py > train.log$i 2>&1 &
    cd ../
done
