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
"""Create train or eval dataset."""
import os
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


device_id = int(os.getenv('DEVICE_ID', '0'))
device_num = int(os.getenv('RANK_SIZE'))


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32):
    """
    Create a train or eval dataset.

    Args:
        dataset_path (str): The path of dataset.
        do_train (bool): Whether dataset is used for train or eval.
        repeat_num (int): The repeat times of dataset. Default: 1.
        batch_size (int): The batch size of dataset. Default: 32.

    Returns:
        Dataset.
    """
    if do_train:
        dataset_path = os.path.join(dataset_path, 'train')
        do_shuffle = True
    else:
        dataset_path = os.path.join(dataset_path, 'eval')
        do_shuffle = False

    if device_num == 1 or not do_train:
        dataset = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=do_shuffle)
    else:
        dataset = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=do_shuffle,
                                    num_shards=device_num, shard_id=device_id)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_flip_op = C.RandomHorizontalFlip(device_id / (device_id + 1))

    resize_op = C.Resize((resize_height, resize_width))
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    change_swap_op = C.HWC2CHW()

    trans = []
    if do_train:
        trans += [random_crop_op, random_horizontal_flip_op]

    trans += [resize_op, rescale_op, normalize_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    dataset = dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=8)

    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    dataset = dataset.repeat(repeat_num)

    return dataset
