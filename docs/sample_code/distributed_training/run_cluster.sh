#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH RANK_TABLE_FILE RANK_SIZE RANK_START"
echo "For example: bash run.sh /path/dataset /path/rank_table.json 16 0"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

execute_path=$(pwd)
echo ${execute_path}
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
echo ${self_path}

export DATA_PATH=$1
export RANK_TABLE_FILE=$2
export RANK_SIZE=$3
RANK_START=$4
DEVICE_START=0
for((i=0;i<=7;i++));
do
  export RANK_ID=$[i+RANK_START]
  export DEVICE_ID=$[i+DEVICE_START]
  rm -rf ${execute_path}/device_$RANK_ID
  mkdir ${execute_path}/device_$RANK_ID
  cd ${execute_path}/device_$RANK_ID || exit
  pytest -s ${self_path}/resnet50_distributed_training.py >train$RANK_ID.log 2>&1 &
done