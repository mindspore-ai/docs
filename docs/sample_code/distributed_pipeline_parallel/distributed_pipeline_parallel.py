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

"""Distributed Pipeline Parallel Example"""

import os
import math
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops, Parameter
from mindspore.communication import init
from mindspore.common.initializer import initializer, HeUniform


ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2)
init()
ms.set_seed(1)

class MatMulCell(nn.Cell):
    """
    MatMulCell definition.
    """
    def __init__(self, param=None, shape=None):
        super().__init__()
        if shape is None:
            shape = [28 * 28, 512]
        weight_init = HeUniform(math.sqrt(5))
        self.param = Parameter(initializer(weight_init, shape), name="param")
        if param is not None:
            self.param = param
        self.print = ops.Print()
        self.matmul = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        self.print("out is:", out)
        return out


class Network(nn.Cell):
    """
    Network definition.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = MatMulCell()
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
net.layer1.pipeline_stage = 0
net.relu1.pipeline_stage = 0
net.layer2.pipeline_stage = 0
net.relu2.pipeline_stage = 1
net.layer3.pipeline_stage = 1

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

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

net_with_loss = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
net_with_loss.set_train()

def forward_fn(inputs, target):
    loss = net_with_loss(inputs, target)
    return loss

grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
pp_grad_reducer = nn.PipelineGradReducer(optimizer.parameters)

@ms.jit
def train_one_step(inputs, target):
    loss, grads = grad_fn(inputs, target)
    grads = pp_grad_reducer(grads)
    optimizer(grads)
    return loss, grads

for epoch in range(10):
    i = 0
    for data, label in data_set:
        loss_value, grads_value = train_one_step(data, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_value))
        i += 1
