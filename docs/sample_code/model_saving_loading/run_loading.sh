#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_loading.sh"
echo "=============================================================================================================="

mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout python test_loading.py
