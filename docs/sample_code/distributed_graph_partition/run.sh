#!/bin/bash
# Launch scheduler and worker for distributed graph partition.
execute_path=$(pwd)

if [ ! -d "${execute_path}/MNIST_Data" ]; then
    if [ ! -f "${execute_path}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi

self_path=$(dirname $0)

# Set public environment.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export DATA_PATH=${execute_path}/MNIST_Data/

# Launch scheduler.
export MS_ROLE=MS_SCHED
rm -rf ${execute_path}/sched/
mkdir ${execute_path}/sched/
cd ${execute_path}/sched/ || exit
python ${self_path}/../train.py > sched.log 2>&1 &

# Launch workers.
export MS_ROLE=MS_WORKER
for((i=0;i<$MS_WORKER_NUM;i++));
do
  rm -rf ${execute_path}/worker_$i/
  mkdir ${execute_path}/worker_$i/
  cd ${execute_path}/worker_$i/ || exit
  python ${self_path}/../train.py > worker_$i.log 2>&1 &
done
echo "Start training. The output is saved in sched and worker_* folder."