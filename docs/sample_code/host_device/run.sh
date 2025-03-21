#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh"
echo "=============================================================================================================="

EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

export GLOG_v=1
msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10801 --join=True \
--log_dir=./log_output pytest -s train.py
