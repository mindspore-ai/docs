#!/bin/bash
# applicable to Ascend

EXEC_PATH=$(pwd)

export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
export RANK_SIZE=8

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./distributed_training.py ./dataset.py ./net.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    pytest -sv ./distributed_training.py::test_train > train.log$i 2>&1 &
    cd ../
done

rm -rf device0
mkdir device0
cp ./distributed_training.py ./dataset.py ./net.py ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
pytest -sv ./distributed_training.py::test_train > train.log0 2>&1
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../

for((i=1;i<${RANK_SIZE};i++))
do
    cp ./distributed_inference.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start inference for device $i"
    pytest -sv ./distributed_inference.py::test_inference > inference.log$i 2>&1 &
    cd ../
done

cp ./distributed_inference.py ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start inference for device 0"
pytest -sv ./distributed_inference.py::test_inference > inference.log0 2>&1
if [ $? -eq 0 ];then
    echo "inference success"
else
    echo "inference failed"
    exit 2
fi
cd ../
