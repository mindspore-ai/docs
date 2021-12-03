#!/bin/bash
# applicable to GPU

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_gpu.sh DATA_PATH DEVICE_TARGET"
echo "DEVICE_TARGET could be GPU or Ascend"
echo "For example: bash run_gpu.sh /path/dataset GPU"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
DEVICE_TARGET=$2
export DATA_PATH=${DATA_PATH}
export DEVICE_TARGET=${DEVICE_TARGET}

if [ "${DEVICE_TARGET}" = "GPU" ]; then
  rm -rf pipeline_gpu
  mkdir pipeline_gpu
  cp ./resnet50_distributed_training_pipeline.py ./resnet.py ./pipeline_gpu
  cd ./pipeline_gpu
  echo "start training"
  mpirun -n 8 pytest -s -v ./resnet50_distributed_training_pipeline.py >train.log 2>&1 &
fi

if [ "${DEVICE_TARGET}" = "Ascend" ]; then
  EXEC_PATH=$(pwd)

  test_dist_8pcs() {
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
  }

  test_dist_8pcs

  for ((i = 1; i < ${RANK_SIZE}; i++)); do
    rm -rf device$i
    mkdir device$i
    cp ./resnet50_distributed_training_pipeline.py ./resnet.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env >env$i.log
    pytest -s -v ./resnet50_distributed_training_pipeline.py >train.log$i 2>&1 &
    cd ../
  done
  rm -rf device0
  mkdir device0
  cp ./resnet50_distributed_training_pipeline.py ./resnet.py ./device0
  cd ./device0
  export DEVICE_ID=0
  export RANK_ID=0
  echo "start training for device 0"
  env >env0.log
  pytest -s -v ./resnet50_distributed_training_pipeline.py >train.log0 2>&1
  if [ $? -eq 0 ]; then
    echo "training success"
  else
    echo "training failed"
    exit 2
  fi
  cd ../
fi