#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_stage_1.sh [DATA_PATH]"
echo "For example: bash run_stage_1.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_PATH=$(get_real_path $1)

EXEC_PATH=$(pwd)

export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
export RANK_SIZE=8
export DEVICE_NUM=8

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device${i}_stage_1
    mkdir device${i}_stage_1
    cp ./train_stage_1.py ./resnet.py ./device${i}_stage_1
    cd ./device${i}_stage_1
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env.log
    python ./train_stage_1.py --data_path=$DATA_PATH > train.log 2>&1 &
    cd ../
done
