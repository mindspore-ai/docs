#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_parallel_save_ckpt.sh DATA_PATH"
echo "For example: bash run.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
export GROUP_INFO_FILE=./group_info.pb
RANK_SIZE=8

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_${RANK_SIZE}pcs

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf ckpt_dir$i
    mkdir ckpt_dir$i
    cp ./parallel_save_ckpt_train.py ./model.py ./dataset.py ./ckpt_dir$i
    cd ./ckpt_dir$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for ckpt_dir $i"
    env > env$i.log
    python ./parallel_save_ckpt_train.py --distribute=true --file_path=${DATA_PATH} --mp=4 --enable_parallel_optimizer=0 > train.log$i 2>&1 &
    cd ../
done
rm -rf ckpt_dir0
mkdir ckpt_dir0
cp ./parallel_save_ckpt_train.py ./model.py ./dataset.py ./ckpt_dir0
cd ./ckpt_dir0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for ckpt_dir 0"
env > env0.log
python ./parallel_save_ckpt_train.py --distribute=true --file_path=${DATA_PATH} --mp=4 --enable_parallel_optimizer=0 > train.log0 2>&1 &
cd ../
