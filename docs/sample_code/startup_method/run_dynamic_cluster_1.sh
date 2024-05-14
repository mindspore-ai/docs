#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_dynamic_cluster_1.sh"
echo "==========================================="

EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf device
mkdir device
echo "start training"

# 循环启动Worker1到Worker4，4个Worker训练进程
for((i=0;i<4;i++));
do
    export MS_WORKER_NUM=8                     # 设置集群中Worker进程数量为8（包括其他节点进程）
    export MS_SCHED_HOST=<node_1 ip address>   # 设置Scheduler IP地址为节点1 IP地址
    export MS_SCHED_PORT=8118                  # 设置Scheduler端口
    export MS_ROLE=MS_WORKER                   # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i                       # 设置进程id，可选
    python ./net.py > device/worker_$i.log 2>&1 &     # 启动训练脚本
done

# 在节点1启动1个Scheduler进程
export MS_WORKER_NUM=8                     # 设置集群中Worker进程总数为8（包括其他节点进程）
export MS_SCHED_HOST=<node_1 ip address>   # 设置Scheduler IP地址为节点1 IP地址
export MS_SCHED_PORT=8118                  # 设置Scheduler端口
export MS_ROLE=MS_SCHED                    # 设置启动的进程为MS_SCHED角色
python ./net.py > device/scheduler.log 2>&1 &     # 启动训练脚本
