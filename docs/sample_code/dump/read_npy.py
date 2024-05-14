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

"""This is a tool to read data from the '.npy' tensor files."""
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--npy_files', type=str, help='the path of npy files')
args = parser.parse_args()
npy_files = args.npy_files
files = os.listdir(npy_files)
l = len(files)
max_read_number = 10
if l < max_read_number:
    max_read_number = l
# Read the first max_read_number files,
for i in range(max_read_number):
    file_name = files[i]
    if not file_name.endswith('.npy'):
        continue
    absolute_path = os.path.join(npy_files, file_name)
    tensor = np.load(absolute_path)
    # At here, the tensor is loaded, you can analyse it now.
    print("Load the file: {} successfully.".format(absolute_path))
