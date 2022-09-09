#!/bin/bash
# run_gpu_cluster.sh

DATA_PATH=$1
HOSTFILE=$2

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py mpirun_gpu_cluster.sh ./device
cd ./device

echo "start training"
mpirun -n 4 --mca btl tcp,self --mca btl_tcp_if_include 10.145.87.0/24 --hostfile $HOSTFILE -x DATA_PATH=$DATA_PATH -x PATH -mca pml ob1 mpirun_gpu_cluster.sh &
