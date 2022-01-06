#!/bin/bash
# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 4 ] && [ $# != 5 ]
then
  echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]"
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $3)
PATH2=$(get_real_path $4)

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -d $PATH2 ]
then 
    echo "error: DATASET_PATH=$PATH2 is not a directory"
exit 1
fi 

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=${i}
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py --run_distribute=True  --dataset_path=$PATH2 &> log &
    cd ..
done
