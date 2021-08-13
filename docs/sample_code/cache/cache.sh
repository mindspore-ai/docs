#!/bin/bash
# This shell script will launch parallel pipelines
# applicable to Ascend

# get path to dataset directory
if [ $# != 1 ]
then
        echo "Usage: sh cache.sh DATASET_PATH"
exit 1
fi
dataset_path=$1

# generate a session id that these parallel pipelines can share
result=$(cache_admin -g 2>&1)
rc=$?
if [ $rc -ne 0 ]; then
    echo "some error"
    exit 1
fi

# grab the session id from the result string
session_id=$(echo $result | awk '{print $NF}')

# make the session_id available to the python scripts
num_devices=4

for p in $(seq 0 $((${num_devices}-1))); do
    python my_training_script.py --num_devices "$num_devices" --device "$p" --session_id $session_id --dataset_path $dataset_path
done
