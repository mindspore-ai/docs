#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash recover.sh"
echo "=============================================================================================================="

export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/
export MS_ENABLE_RECOVERY=1                # 开启容灾功能
export MS_RECOVERY_PATH=/path/to/recovery/ # 设置容灾文件保存路径

# 启动1个Scheduler进程
export MS_WORKER_NUM=8              # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=8118           # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
export MS_NODE_ID=sched             # 设置本节点Node ID为'sched'
python ./train.py > device/scheduler.log 2>&1 &     # 启动训练脚本
