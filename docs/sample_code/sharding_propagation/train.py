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
"""Sharding Propagation Programming Guide
This sample code is applicable to Ascend.
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter, set_context, GRAPH_MODE, set_auto_parallel_context
from mindspore.nn import Cell, Momentum
import mindspore.ops as ops
from mindspore import Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits
import mindspore.dataset as ds
import mindspore.communication as D
from mindspore import LossMonitor
from mindspore import ModelCheckpoint


step_per_epoch = 4

def get_dataset(*inputs):
    def generate():
        for _ in range(step_per_epoch):
            yield inputs
    return generate

class Dense(Cell):
    """Dense layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones([in_channels, out_channels]), dtype=ms.float32), name="weight1")
        self.bias = Parameter(Tensor(np.ones([out_channels]), dtype=ms.float32), name="bias")
        self.matmul = ops.MatMul()
        self.add = ops.Add()

    def construct(self, x):
        x = self.matmul(x, self.weight)
        x = self.add(x, self.bias)
        return x

class FFN(Cell):
    """FeedForward Network"""
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(64, 64)
        self.dense1.matmul.shard(((2, 1), (1, 4)))
        self.relu = ops.ReLU()
        self.dense2 = Dense(64, 64)

    def construct(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    set_context(mode=GRAPH_MODE, device_target="Ascend", save_graphs=True)
    D.init()
    rank = D.get_rank()
    set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                              device_num=8, full_batch=True)

    np.random.seed(1)
    input_data = np.random.rand(64, 64).astype(np.float32)
    label_data = np.random.rand(64, 64).astype(np.float32)

    fake_dataset = get_dataset(input_data, label_data)
    net = FFN()

    learning_rate = 0.001
    momentum = 0.1
    epoch_size = 1

    callback = [LossMonitor(), ModelCheckpoint(directory="{}".format(rank))]

    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])
    loss = SoftmaxCrossEntropyWithLogits()
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=callback, dataset_sink_mode=False)
