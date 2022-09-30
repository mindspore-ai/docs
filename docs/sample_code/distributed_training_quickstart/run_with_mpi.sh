#!/bin/bash
# applicable to GPU

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_with_mpi.sh"
echo "=============================================================================================================="
set -e

rm -rf device saved_graph
mkdir device
cp ./net.py ./device
cd ./device
echo "start training"
mpirun --allow-run-as-root -n 8 python ./net.py > train.log 2>&1 &