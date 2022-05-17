#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_gpu.sh DATA_PATH"
echo "For example: bash run_with_mpi.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
export DATA_PATH=${DATA_PATH:-$1}

rm -rf device
mkdir device
cp ./resnet50_distributed_training.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout pytest -s -v ./resnet50_distributed_training.py > train.log 2>&1 &