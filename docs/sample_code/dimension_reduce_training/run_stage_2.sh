#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_stage_2.sh [DATA_PATH] [PRETRAINED_WEIGHT_PATH] [PCA_MAT_PATH](optional)"
echo "For example: bash run_stage_2.sh /path/dataset /path/checkpoint_first/resnet-70_625.ckpt /path/pca_mat.npy"
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
PRETRAINED_WEIGHT_PATH=$(get_real_path $2)
if [ $# == 3 ]
then
    PCA_MAT_PATH=$(get_real_path $3)
fi

EXEC_PATH=$(pwd)

export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
export RANK_SIZE=8
export DEVICE_NUM=8

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device${i}_stage_2
    mkdir device${i}_stage_2
    cp ./train_boost_stage_2.py ./resnet.py ./device${i}_stage_2
    cd ./device${i}_stage_2
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log

    if [ $# == 2 ]
    then
        python ./train_boost_stage_2.py --data_path=$DATA_PATH --pretrained_weight_path=$PRETRAINED_WEIGHT_PATH \
        > train.log 2>&1 &
    fi

    if [ $# == 3 ]
    then
        python ./train_boost_stage_2.py --data_path=$DATA_PATH --pretrained_weight_path=$PRETRAINED_WEIGHT_PATH \
        --pca_mat_path=$PCA_MAT_PATH > train.log 2>&1 &
    fi

    cd ../
done
