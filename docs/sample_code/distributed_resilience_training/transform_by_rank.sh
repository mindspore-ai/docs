#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash transform_by_rank.sh DATA_PATH"
echo "For example: bash run.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
RANK_SIZE=4
EXEC_PATH=$(pwd)

SRC_STRATEGY_FILE=$1
DST_STRATEGY_FILE=$2
SRC_CHECKPOINTS_DIR=$3
DST_CHECKPOINTS_DIR=$4
rm -rf $DST_CHECKPOINTS_DIR
mkdir $DST_CHECKPOINTS_DIR

for((i=0;i<${RANK_SIZE};i++))
do
    mkdir $DST_CHECKPOINTS_DIR/rank_$i
    echo "start transforming for rank_$i"
    python ./transform_checkpoint_by_rank.py --transform_rank=$i --src_strategy_file=${SRC_STRATEGY_FILE} --dst_strategy_file=${DST_STRATEGY_FILE} \
    --src_checkpoints_dir=${SRC_CHECKPOINTS_DIR} --dst_checkpoints_dir=${DST_CHECKPOINTS_DIR} > transform_rank$i.log 2>&1 &
done
