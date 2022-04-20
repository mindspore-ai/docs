#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

# This script is used to run the async dump on Ascend.

# Set the dump config file.
export MINDSPORE_DUMP_CONFIG="./async_dump.json"
export npy_tensor_path="./npy_tensor_path"
export run_install_path="/usr/local/Ascend"

# Set the device target and device id.
export DEVICE_TARGET="Ascend"
export DEVICE_ID=0

# Run the training script.
python ./train_alexnet.py --device_target=${DEVICE_TARGET} --device_id=${DEVICE_ID} > train_log.txt 2>&1 &&

# Get the tensor path
tensor_path=$(python get_tensor_path.py --dump_config ${MINDSPORE_DUMP_CONFIG})
# Convert the bin files to npy files. If you set the file_format as 'npy', this step should be skipped.
absolute_convert_path=$(find ${run_install_path} -name "msaccucmp.py")
python ${absolute_convert_path} "convert" -d ${tensor_path} -out ${npy_tensor_path} -f NCHW -t npy

# Read the converted npy files.
python read_npy.py --npy_files ${npy_tensor_path} > read_tensor_log.txt 2>&1 &
