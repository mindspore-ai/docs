#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_infer.sh"
echo "=============================================================================================================="

mpirun -n 4 --output-filename log_output --merge-stderr-to-stdout python model_transformation_infer.py
