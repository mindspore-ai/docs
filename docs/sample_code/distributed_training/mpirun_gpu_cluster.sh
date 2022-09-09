#!/bin/bash
# mpirun_gpu_cluster.sh

export NCCL_SOCKET_IFNAME="eno1"

pytest -s -v ./resnet50_distributed_training_gpu.py > train.log 2>&1 
