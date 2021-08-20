# Copyright 2020 Huawei Technologies Co., Ltd
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
'''
Dataset
'''
import numpy as np
from mindspore import Tensor
from mindspore.communication import init, get_rank, get_group_size


class FakeData:
    """custom dataset"""
    def __init__(self, size=256, batch_size=16, image_size=(96,), num_classes=16, random_offset=0):
        """init"""
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_epochs = -1
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = np.float32
        self.label_data_type = np.float32
        self.is_onehot = True
        init(backend_name='hccl')
        self.rank_size = get_group_size()
        self.rank_id = get_rank()
        self.total_batch_size = self.rank_batch_size * self.rank_size
        self.total_batch_data_size = (self.rank_size, self.rank_batch_size) + image_size
        self.do_copy = False

    def get_dataset_size(self):
        """get dataset size"""
        return int(self.size / self.total_batch_size)

    def get_repeat_count(self):
        """get repeat count"""
        return 1

    def create_tuple_iterator(self, num_epochs=-1, do_copy=False):
        """create tuple iterator"""
        self.num_epochs = num_epochs
        self.do_copy = do_copy
        return self

    def __getitem__(self, batch_index):
        """get item"""
        if batch_index * self.total_batch_size >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        img = np.random.randn(*self.total_batch_data_size)
        target = np.random.randint(0, self.num_classes, size=(self.rank_size, self.rank_batch_size))
        np.random.set_state(rng_state)
        img = img[self.rank_id]
        target = target[self.rank_id]
        img_ret = img.astype(self.image_data_type)
        target_onehot = np.zeros(shape=(self.rank_batch_size, self.num_classes))
        target_onehot[np.arange(self.rank_batch_size), target] = 1
        target_ret = target_onehot.astype(self.label_data_type)
        return Tensor(img_ret), Tensor(target_ret)

    def __len__(self):
        """get size"""
        return self.size

    def __iter__(self):
        """iter"""
        self.batch_index = 0
        return self

    def reset(self):
        """reset index"""
        self.batch_index = 0

    def __next__(self):
        """next"""
        if self.batch_index * self.total_batch_size < len(self):
            data = self[self.batch_index]
            self.batch_index += 1
            return data
        raise StopIteration
