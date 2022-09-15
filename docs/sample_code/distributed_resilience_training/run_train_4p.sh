#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_train_4p.sh DATA_PATH"
echo "For example: bash run.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
RANK_SIZE=4

EXEC_PATH=$(pwd)

test_dist_4pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_4pcs.json
    export RANK_SIZE=4
}

test_dist_${RANK_SIZE}pcs

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf recover$i
    mkdir recover$i
    cp ./train_4p.py ./model.py ./dataset.py ./recover$i
    cd ./recover$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for recover $i"
    env > env$i.log
    python ./train_4p.py --distribute=true --file_path=${DATA_PATH} --mp=4 \
    --enable_parallel_optimizer=0 --ckpt_file=../dst_checkpoints/rank_$i/transformed$i.ckpt > train.log$i 2>&1 &
    cd ../
done
rm -rf recover0
mkdir recover0
cp ./train_4p.py ./model.py ./dataset.py ./recover0
cd ./recover0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for recover 0"
env > env0.log
python ./train_4p.py --distribute=true --file_path=${DATA_PATH} --mp=4 \
--enable_parallel_optimizer=0 --ckpt_file=../dst_checkpoints/rank_0/transformed0.ckpt > train.log0 2>&1 &
cd ../
