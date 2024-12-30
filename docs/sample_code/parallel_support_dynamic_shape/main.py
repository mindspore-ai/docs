# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Distributed parallel support dynamic shape example."""
import os
import mindspore as ms
import mindspore.dataset as ds
import mindspore.runtime as rt
from mindspore import nn, ops, Model
from mindspore import Symbol, Tensor, Parameter
from mindspore.communication import init
from mindspore.common.initializer import initializer
from mindspore.train import LossMonitor

ms.set_context(mode=ms.GRAPH_MODE)
rt.set_memory(max_size="28GB")
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()
ms.set_seed(1)


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


class Network(nn.Cell):
    """Network"""

    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = Parameter(initializer("normal", [28 * 28, 512], ms.float32))
        self.fc2_weight = Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul().shard(((2, 4), (4, 1)))
        self.relu1 = ops.ReLU().shard(((4, 1),))
        self.matmul2 = ops.MatMul().shard(((1, 8), (8, 1)))
        self.relu2 = ops.ReLU().shard(((8, 1),))
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = ops.reshape(x, (-1, 784))
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        return self.matmul3(x, self.fc3_weight)


net = Network()

data_set = create_dataset(32)  # (32, 1, 28, 28) (32,)
optimizer = nn.SGD(net.trainable_params(), 1e-3)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(True)

s0 = Symbol(divisor=8)
input_dyn = Tensor(shape=[s0, 1, 28, 28], dtype=ms.float32)
label_dyn = Tensor(shape=[s0], dtype=ms.int32)
net.set_inputs(input_dyn)
loss_fn.set_inputs(input_dyn, label_dyn)

model = Model(net, loss_fn, optimizer)
model.train(5, data_set, callbacks=[LossMonitor()], dataset_sink_mode=False)
