#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh [DATA_PATH]"
echo "For example: bash run.sh /path/dataset"
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

export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_16pcs.json
export RANK_SIZE=16
export DEVICE_NUM=8

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0;i<${DEVICE_NUM};i++))
do
    rm -rf device${i}
    mkdir device${i}
    cp ./train.py ./resnet.py ./device${i}
    cd ./device${i}
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    echo "start training for device $i"
    env > env.log
    python ./train.py --data_path=$DATA_PATH > train.log 2>&1 &
    cd ../
done
