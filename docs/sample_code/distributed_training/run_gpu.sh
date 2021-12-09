#!/bin/bash
# applicable to GPU

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_gpu.sh DATA_PATH"
echo "For example: bash run_gpu.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 8 pytest -s -v ./resnet50_distributed_training_gpu.py > train.log 2>&1 &
