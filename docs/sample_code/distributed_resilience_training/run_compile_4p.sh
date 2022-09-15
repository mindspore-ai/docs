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
    rm -rf compile$i
    mkdir compile$i
    cp ./train_4p.py ./model.py ./dataset.py ./compile$i
    cd ./compile$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for compile $i"
    env > env$i.log
    python ./train_4p.py --distribute=true --file_path=${DATA_PATH} --mp=4 \
    --enable_parallel_optimizer=0 --only_compile=1 > train.log$i 2>&1 &
    cd ../
done
rm -rf compile0
mkdir compile0
cp ./train_4p.py ./model.py ./dataset.py ./compile0
cd ./compile0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for compile 0"
env > env0.log
python ./train_4p.py --distribute=true --file_path=${DATA_PATH} --mp=4 \
--enable_parallel_optimizer=0 --only_compile=1 > train.log0 2>&1 &
cd ../
