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
"""resnet dataset functions."""
import os
import multiprocessing
import mindspore as ms
import mindspore.dataset as ds
ms_version = ms.__version__.split(".")
if int(ms_version[0]) == 1 and int(ms_version[1]) <= 7:
    from mindspore.dataset.vision import c_transforms as vision
    from mindspore.dataset.transforms.c_transforms import TypeCast
else:
    from mindspore.dataset import vision
    from mindspore.dataset.transforms.transforms import TypeCast


def create_imagenet_dataset(dataset_path, do_train, batch_size=32, image_size=(224, 224), rank_size=1, rank_id=0):
    """
    create a imagenet dataset for resnet
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32
        image_size(tuple[int]): the image size of input image. Default: 224
        rank_size (int): Number of shards that the dataset will be divided. Default: 1
        rank_id (int): The shard ID within num_shards (default=None). Default: 0

    Returns:
        dataset
    """
    dataset = ds.ImageFolderDataset(dataset_path, num_shards=rank_size, shard_id=rank_id, shuffle=do_train,
                                    num_parallel_workers=get_num_parallel_workers(12))

    # define map operations
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if do_train:
        trans = [
            vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
    else:
        resize_ratio = 256 / 224
        resize_size = (round(image_size[0] * resize_ratio), round(image_size[1] * resize_ratio))
        trans = [
            vision.Decode(),
            vision.Resize(resize_size),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=get_num_parallel_workers(8))

    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=do_train)
    return dataset


def create_cifar_dataset(dataset_path, do_train, batch_size=32, image_size=(224, 224), rank_size=1, rank_id=0):
    """
    create a cifar10 dataset for resnet
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32
        image_size(tuple[int]): the image size of input image. Default: 224
        rank_size (int): Number of shards that the dataset will be divided. Default: 1
        rank_id (int): The shard ID within num_shards (default=None). Default: 0

    Returns:
        dataset
    """
    dataset = ds.Cifar10Dataset(dataset_path, num_parallel_workers=get_num_parallel_workers(12), shuffle=do_train,
                                num_shards=rank_size, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        vision.Resize(image_size),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]

    type_cast_op = TypeCast(ms.int32)

    data_set = dataset.map(operations=type_cast_op, input_columns="label",
                           num_parallel_workers=get_num_parallel_workers(8))
    data_set = data_set.map(operations=trans, input_columns="image",
                            num_parallel_workers=get_num_parallel_workers(8))

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=do_train)
    return data_set


def create_dataset(dataset, dataset_path, do_train, batch_size=32, image_size=(224, 224), rank_size=1, rank_id=0):
    """
    create a dataset for resnet, support imagenet and cifar10
    Args:
        dataset(string): dataset mode, support imagenet and cifar10.
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32
        image_size(tuple[int]): the image size of input image. Default: 224
        rank_size (int): Number of shards that the dataset will be divided. Default: 1
        rank_id (int): The shard ID within num_shards (default=None). Default: 0

    Returns:
        dataset
    """
    if (not isinstance(dataset, str)) and (dataset not in ["imagenet", "cifar10"]):
        raise ValueError(f"dataset in create_dataset mast be in ['imagenet', 'cifar10'], but get {dataset}")
    if not os.path.isdir(dataset_path):
        raise ValueError(f"dataset_path: {dataset_path} in create_dataset is not a directory")
    if dataset == "imagenet":
        if do_train and os.path.isdir(os.path.join(dataset_path, "train")):
            return create_imagenet_dataset(os.path.join(dataset_path, "train"), do_train, batch_size=batch_size,
                                           image_size=image_size, rank_size=rank_size, rank_id=rank_id)
        if (not do_train) and os.path.isdir(os.path.join(dataset_path, "val")):
            return create_imagenet_dataset(os.path.join(dataset_path, "val"), do_train, batch_size=batch_size,
                                           image_size=image_size, rank_size=rank_size, rank_id=rank_id)
        raise ValueError(f"There must be 'train' and 'val' directories under the dataset_path: {dataset_path}\n"
                         f"like:\n"
                         f"└─dataset_path\n"
                         f"    ├─train     # train dataset\n"
                         f"    └─val       # evaluate dataset\n")
    if do_train and os.path.isdir(os.path.join(dataset_path, "cifar-10-batches-bin")):
        return create_cifar_dataset(os.path.join(dataset_path, "cifar-10-batches-bin"),
                                    do_train, batch_size=batch_size,
                                    image_size=image_size, rank_size=rank_size, rank_id=rank_id)
    if (not do_train) and os.path.isdir(os.path.join(dataset_path, "cifar-10-verify-bin")):
        return create_cifar_dataset(os.path.join(dataset_path, "cifar-10-verify-bin"),
                                    do_train, batch_size=batch_size,
                                    image_size=image_size, rank_size=rank_size, rank_id=rank_id)
    raise ValueError(f"There must be 'cifar-10-batches-bin' and 'cifar-10-verify-bin' directories "
                     f"under the dataset_path: {dataset_path}\n"
                     f"like:\n"
                     f"└─dataset_path\n"
                     f"    ├─cifar-10-batches-bin      # train dataset\n"
                     f"    └─cifar-10-verify-bin       # evaluate dataset\n")


def get_num_parallel_workers(num_parallel_workers, rank_size=1):
    """
    Get num_parallel_workers used in dataset operations.
    When run standalone, if num_parallel_workers > the real CPU cores number,
    set num_parallel_workers = the real CPU cores number.
    When run distribute, for improving CPU utilization and reduce competition,
    set num_parallel_workers = the real CPU cores number // rank_size, The maximum rank_size is 8.
    """
    # Generally, a server can have up to 8 cards.
    rank_size = 8 if rank_size > 8 else rank_size
    cores = multiprocessing.cpu_count() // rank_size
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
            num_parallel_workers = cores
    else:
        print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 8)))
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers
