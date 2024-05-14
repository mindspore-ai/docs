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
"""PyNative Shard Function Parallel Example"""
import os
import mindspore as ms
from mindspore import nn
import mindspore.dataset as ds
from mindspore.communication import init

ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL,
                             search_mode="sharding_propagation", device_num=8)
init()
ms.set_seed(1)

class BasicBlock(nn.Cell):
    """A base layer with two dense layer"""
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.dense1 = nn.Dense(128, 256)
        self.gelu = nn.GELU()
        self.dense2 = nn.Dense(256, 128)
    def construct(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return x

class Net(nn.Cell):
    """A network with three basicblock"""
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = BasicBlock()
        self.block2 = BasicBlock()
        self.block3 = BasicBlock()
    def construct(self, x):
        # 此处所有blocks都在PyNative模式执行
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class Net1(Net):
    """A shard function example"""
    def __init__(self):
        super(Net1, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 128)
        self.layer2 = nn.Dense(128, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        # block1在图模式执行
        x = self.block1(x)
        # block2和block3在PyNative模式执行
        x = self.block2(x)
        x = self.block3(x)
        x = self.layer2(x)
        return x

net = Net1()
# 沿输入第二轴进行切片，使得输出变成数据并行排布
net.block1.shard(in_strategy=((1, 8),), parameter_plan={'self.block1.dense2.weight': (8, 1)})

def create_dataset(batch_size):
    """create dataset"""
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)
optimizer = nn.SGD(net.trainable_params(), 1e-1)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    """forward propagation"""
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

for epoch in range(5):
    i = 0
    for image, label in data_set:
        (loss_value, _), grads = grad_fn(image, label)
        optimizer(grads)
        if i % 100 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_value))
        i += 1

print("=========Finish=========")
