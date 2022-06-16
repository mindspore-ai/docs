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

"""Fix randomness example.
This sample code is applicable to GPU and Ascend.
"""

import os
import shutil
import urllib.request
from urllib.parse import urlparse
import argparse
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
import mindspore.ops as ops
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--device_target', type=str, default='Ascend',
                    help="the device target, it must be one of ['GPU','Ascend', 'CPU'], default is 'Ascend'.")
parser.add_argument('--device_id', type=int, default=0, help="the device id, default is 0.")
args = parser.parse_args()


def callbackfunc(blocknum, blocksize, totalsize):
    percent = 100.0 * blocknum * blocksize / totalsize
    percent = min(percent, 100)
    print("downloaded {:.1f}".format(percent), end="\r")


def _download_dataset():
    ds_url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    file_base_name = urlparse(ds_url).path.split("/")[-1]
    file_name = os.path.join("./datasets", file_base_name)
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(ds_url, file_name, callbackfunc)
    print("{:*^40}".format("DataSets Downloaded"))
    shutil.unpack_archive(file_name, extract_dir="./datasets/cifar-10-binary")


def _copy_dataset(ds_part, dest_path):
    data_source_path = "./datasets/cifar-10-binary/cifar-10-batches-bin"
    ds_part_source_path = os.path.join(data_source_path, ds_part)
    if not os.path.exists(ds_part_source_path):
        _download_dataset()
    shutil.copy(ds_part_source_path, dest_path)


def download_cifar10_dataset():
    """
    Download the cifar10 dataset.
    """
    ds_base_path = "./datasets/cifar10"
    train_path = os.path.join(ds_base_path, "train")
    test_path = os.path.join(ds_base_path, "test")
    print("{:*^40}".format("Checking DataSets Path."))
    if not os.path.exists(train_path) and not os.path.exists(test_path):
        os.makedirs(train_path)
        os.makedirs(test_path)
    print("{:*^40}".format("Downloading CIFAR-10 DataSets."))
    for i in range(1, 6):
        train_part = "data_batch_{}.bin".format(i)
        if not os.path.exists(os.path.join(train_path, train_part)):
            _copy_dataset(train_part, train_path)
        pops = train_part + " is ok"
        print("{:*^40}".format(pops))
    test_part = "test_batch.bin"
    if not os.path.exists(os.path.join(test_path, test_part)):
        _copy_dataset(test_part, test_path)
    print("{:*^40}".format(test_part + " is ok"))
    print("{:*^40}".format("Downloaded CIFAR-10 DataSets Already."))


def create_dataset_cifar10(data_path, batch_size=32, repeat_size=1, status="train"):
    """
    create dataset for train or test
    """
    cifar_ds = ds.Cifar10Dataset(data_path)
    rescale = 1.0 / 255.0
    shift = 0.0

    resize_op = vision.Resize(size=(227, 227))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if status == "train":
        random_crop_op = vision.RandomCrop([32, 32], [4, 4, 4, 4])
        random_horizontal_op = vision.RandomHorizontalFlip()
    channel_swap_op = vision.HWC2CHW()
    typecast_op = transforms.TypeCast(ms.int32)
    cifar_ds = cifar_ds.map(operations=typecast_op, input_columns="label")
    if status == "train":
        cifar_ds = cifar_ds.map(operations=random_crop_op, input_columns="image")
        cifar_ds = cifar_ds.map(operations=random_horizontal_op, input_columns="image")
    cifar_ds = cifar_ds.map(operations=resize_op, input_columns="image")
    cifar_ds = cifar_ds.map(operations=rescale_op, input_columns="image")
    cifar_ds = cifar_ds.map(operations=normalize_op, input_columns="image")
    cifar_ds = cifar_ds.map(operations=channel_swap_op, input_columns="image")

    cifar_ds = cifar_ds.shuffle(buffer_size=1000)
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    cifar_ds = cifar_ds.repeat(repeat_size)
    return cifar_ds


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid"):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode=pad_mode)


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    return TruncatedNormal(0.02)


class AlexNet(nn.Cell):
    """
    Alexnet
    """

    def __init__(self, num_classes=10, channel=3):
        super(AlexNet, self).__init__()
        self.conv1 = conv(channel, 96, 11, stride=4)
        self.conv2 = conv(96, 256, 5, pad_mode="same")
        self.conv3 = conv(256, 384, 3, pad_mode="same")
        self.conv4 = conv(384, 384, 3, pad_mode="same")
        self.conv5 = conv(384, 256, 3, pad_mode="same")
        self.relu = nn.ReLU()
        self.max_pool2d = ops.MaxPool(kernel_size=3, strides=2)
        self.flatten = nn.Flatten()
        self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
        self.fc2 = fc_with_initialize(4096, 4096)
        self.fc3 = fc_with_initialize(4096, num_classes)

    def construct(self, x):
        """
        The construct function.

        Args:
           x(int): Input of the network.

        Returns:
           Tensor, the output of the network.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def get_lr(current_step, lr_max, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       current_step(int): current steps of the training
       lr_max(float): max learning rate
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.8 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr = lr_max
        else:
            lr = lr_max * 0.1
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def train(ds_train):
    """
    the training and evaluation function.

    Args:
       ds_train(mindspore.dataset): The dataset for training.

    Returns:
       None.
    """
    max_step = 10
    device_target = args.device_target
    device_id = args.device_id
    ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target, device_id=device_id)
    network = AlexNet(num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_with_loss = nn.WithLossCell(network, net_loss)
    lr = ms.Tensor(get_lr(0, 0.002, 10, ds_train.get_dataset_size()))
    net_opt = nn.Momentum(net_with_loss.trainable_params(), learning_rate=lr, momentum=0.9)
    # define training net
    train_net = nn.TrainOneStepCell(net_with_loss, net_opt)
    # set the net to train mode
    train_net.set_train()

    step = 1
    for inputs in ds_train:
        output = train_net(*inputs)
        print("step: {0}, losses: {1}".format(step, output.asnumpy()))
        step += 1
        if step > max_step:
            break


if __name__ == "__main__":
    download_cifar10_dataset()
    data_train = create_dataset_cifar10(data_path="./datasets/cifar10/train")
    train(data_train)
