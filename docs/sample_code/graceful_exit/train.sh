#!/bin/bash

echo "==================================================="
echo "Please run the script as: "
echo "bash train.sh"
echo "==================================================="

EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi

export MS_ENABLE_GRACEFUL_EXIT=1
msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10970 --join=True --log_dir=./logs graceful_exit.py
