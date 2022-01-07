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
"""
create train or eval dataset.
"""
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


def create_dataset(dataset_path, batch_size=32, rank_size=1, rank_id=0, do_train=True):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        batch_size(int): the batch size of dataset. Default: 32
        rank_size(int): total num of devices for training. Default: 1,
                        greater than 1 in distributed training
        rank_id(int): logical sequence in all devices. Default: 1,
                      can be greater than i in distributed training
        do_train(bool): whether in train mode or eval mode

    Returns:
        dataset
    """
    # num_paralel_workers: parallel degree of data process
    # num_shards: total number devices for distribute training, which equals number shard of data
    # shard_id: the sequence of current device in all distribute training devices,
    #           which equals the data shard sequence for current device
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=do_train,
                                     num_shards=rank_size, shard_id=rank_id)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    trans = [
        C.Decode(),
        C.Resize(256),
        C.CenterCrop(224),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    # call data operations by map
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=do_train)

    return data_set
