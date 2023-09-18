#!/bin/bash
# applicable to Ascend or GPU

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh BATCH_SIZE MEMORY_OFFLOAD"
echo "For example: bash run.sh 128  ON"
echo "=============================================================================================================="
set -e
EXEC_PATH=$(pwd)
BATCH_SIZE=$1
MEMORY_OFFLOAD=$2
OFFLOAD_PARAM="cpu"
AUTO_OFFLOAD=true
OFFLOAD_CPU_SIZE="512GB"
OFFLOAD_DISK_SIZE="1024GB"

EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/cifar-10-binary" ]; then
    if [ ! -f "${EXEC_PATH}/cifar-10-binary.tar.gz" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz
    fi
    tar -zxvf cifar-10-binary.tar.gz
fi
export DATA_PATH=${EXEC_PATH}/cifar-10-batches-bin

mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout  python train.py \
  --batch_size=$BATCH_SIZE --memory_offload=$MEMORY_OFFLOAD \
  --offload_param=$OFFLOAD_PARAM --auto_offload=$AUTO_OFFLOAD \
  --offload_cpu_size=$OFFLOAD_CPU_SIZE --offload_disk_size=$OFFLOAD_DISK_SIZE \
  --host_mem_block_size="1GB" --enable_pinned_mem=true --enable_aio=true
