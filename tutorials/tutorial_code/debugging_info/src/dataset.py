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
from mindspore import Tensor


def create_train_dataset(image_size=(1, 32, 32), num_classes=10):
    """train dataset."""
    ds = CustomDataSet(image_size=image_size, num_classes=num_classes)
    return ds


def create_eval_dataset(image_size=(1, 32, 32), num_classes=10):
    """eval dataset"""
    ds = CustomDataSet(size=2048, batch_size=2048, image_size=image_size, num_classes=num_classes)
    return ds


class CustomDataSet:
    """CustomDataset"""
    def __init__(self, size=32768, batch_size=32, image_size=(1, 32, 32), num_classes=10, is_onehot=True):
        """init"""
        self.size = size
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_index = 0
        self.is_onehot = is_onehot
        self.repeat_count = 1
        self.batch_data_size = (self.batch_size,) + image_size

    def get_dataset_size(self):
        """get dataset size"""
        return int(self.size / self.batch_size)

    def get_repeat_count(self):
        """get repeat count"""
        return self.repeat_count

    def create_tuple_iterator(self, num_epochs=-1, do_copy=False):
        """create tuple iterator"""
        self.num_epochs = num_epochs
        self.do_copy = do_copy
        return self

    def __getitem__(self, batch_index):
        """get item"""
        if batch_index * self.batch_size >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = np.random.get_state()
        np.random.seed(batch_index)
        img = np.random.randn(*self.batch_data_size)
        target = np.random.randint(0, self.num_classes, size=(1, self.batch_size))
        np.random.set_state(rng_state)
        img_ret = img.astype(np.float32)
        target_ret = target.astype(np.float32)
        if self.is_onehot:
            target_onehot = np.zeros(shape=(self.batch_size, self.num_classes))
            target_onehot[np.arange(self.batch_size), target] = 1
            target_ret = target_onehot.astype(np.float32)
        return Tensor(img_ret), Tensor(target_ret)

    def __len__(self):
        """get size"""
        return self.size

    def __iter__(self):
        """iter dataset"""
        self.batch_index = 0
        return self

    def reset(self):
        """reset dataset"""
        self.batch_index = 0

    def __next__(self):
        """get next batch"""
        if self.batch_index * self.batch_size < len(self):
            data = self[self.batch_index]
            self.batch_index += 1
            return data
        raise StopIteration
