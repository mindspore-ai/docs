#!/bin/bash
# applicable to Ascend or GPU

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DEVICE_TARGET BATCH_SIZE MEMORY_OFFLOAD"
echo "For example: bash run.sh Ascend 512 ON"
echo "=============================================================================================================="
set -e
EXEC_PATH=$(pwd)
DEVICE_TARGET=$1
BATCH_SIZE=$2
MEMORY_OFFLOAD=$3
OFFLOAD_PARAM="cpu"
AUTO_OFFLOAD=true
OFFLOAD_CPU_SIZE="512GB"
OFFLOAD_DISK_SIZE="1024GB"

rm -rf run_train
mkdir run_train
cp  -rf ./cifar_resnet50.py ./resnet.py ./run_train
cd ./run_train
export DEVICE_ID=0
export RANK_ID=0
echo "start training"
env > env.log
python ./cifar_resnet50.py  --device_target=$DEVICE_TARGET --batch_size=$BATCH_SIZE --memory_offload=$MEMORY_OFFLOAD \
  --offload_param=$OFFLOAD_PARAM --auto_offload=$AUTO_OFFLOAD \
  --offload_cpu_size=$OFFLOAD_CPU_SIZE --offload_disk_size=$OFFLOAD_DISK_SIZE \
  --host_mem_block_size="1GB" --enable_pinned_mem=true --enable_aio=true \
  > log.txt 2>&1 &
cd ../