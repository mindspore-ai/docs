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

export MS_ENABLE_TFT='{TTP:1}'
export MINDIO_FOR_MINDSPORE=1

msrun --worker_num=4 --local_worker_num=4 --master_port=10970 --join=False --log_dir=msrun_log --cluster_time_out=300  mindio_ttp_case.py
