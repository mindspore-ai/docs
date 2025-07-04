RANK_SIZE=16
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_cross_cluster_16pcs.json
export RANK_SIZE=$RANK_SIZE

RANK_START=$1
DEVICE_START=0

for((i=0;i<=7;i++));
do
  export RANK_ID=$[i+RANK_START]
  export DEVICE_ID=$[i+DEVICE_START]
  rm -rf ./device_$RANK_ID
  mkdir ./device_$RANK_ID
  cp ./allgather_test.py ./device_$RANK_ID
  cd ./device_$RANK_ID
  env > env$i.log
  python ./allgather_test.py >train$RANK_ID.log 2>&1 &
done
