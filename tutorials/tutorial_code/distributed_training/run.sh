#!/bin/bash

EXEC_PATH=$(pwd)
export MINDSPORE_HCCL_CONFIG_PATH=${EXEC_PATH}/rank_table.json
export RANK_SIZE=8

for((i=0;i<$RANK_SIZE;i++))
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
