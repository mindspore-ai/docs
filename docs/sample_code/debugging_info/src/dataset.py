# Copyright 2021 Huawei Technologies Co., Ltd
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
"""dataset
Custom dataset.
"""
import numpy as np
from mindspore import dataset as ds

def get_data(num, img_size=(1, 32, 32), num_classes=10, is_onehot=True):
    for _ in range(num):
        img = np.random.randn(*img_size)
        target = np.random.randint(0, num_classes)
        target_ret = np.array([target]).astype(np.float32)
        if is_onehot:
            target_onehot = np.zeros(shape=(num_classes,))
            target_onehot[target] = 1
            target_ret = target_onehot.astype(np.float32)
        yield img.astype(np.float32), target_ret

def create_train_dataset(num_data=32768, batch_size=32, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data

def create_eval_dataset(num_data=2048, batch_size=2048, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data
