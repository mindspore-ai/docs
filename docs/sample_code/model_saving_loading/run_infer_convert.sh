#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_infer_convert.sh"
echo "=============================================================================================================="

rm -rf dst_checkpoints
mkdir dst_checkpoints

for((i=0;i<4;i++));
do
    mkdir dst_checkpoints/rank_$i
done

mpirun -n 4 --output-filename log_output --merge-stderr-to-stdout python model_transformation_infer.py --only_compile=1
