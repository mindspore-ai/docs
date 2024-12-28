# Copyright 2023 Huawei Technologies Co., Ltd
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
"""LeNet."""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, load_checkpoint, load_param_into_net
from mindspore.common.initializer import One


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """
    def __init__(self, num_class=1, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, weight_init='ones', pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, weight_init='ones', pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=One())
        self.fc2 = nn.Dense(120, 84, weight_init=One())
        self.fc3 = nn.Dense(84, num_class, weight_init=One())

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def mindspore_running(ckpt_path):
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_device(device_target="GPU")
    np_in = Tensor(np.ones([8, 1, 32, 32]).astype(np.float32))
    network = LeNet5()
    params_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, params_dict)
    outs = network(np_in)

    return outs.asnumpy()

if __name__ == '__main__':
    ckpt_dir = './tf2mindspore.ckpt'
    mindspore_running(ckpt_dir)
