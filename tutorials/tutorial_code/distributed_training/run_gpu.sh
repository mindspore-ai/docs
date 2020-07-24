#!/bin/bash

DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training.py ./resnet.py ./device
cd ./device
echo "start training"
pytest -s -v ./resnet50_distributed_training.py > train.log 2>&1 &
