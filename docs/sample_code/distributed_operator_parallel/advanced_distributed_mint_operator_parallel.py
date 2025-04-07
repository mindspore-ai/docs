# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Distributed mint Operator Parallel Example"""

import os
import mindspore as ms
import mindspore.dataset as ds
import mindspore.runtime as rt
from mindspore import nn, mint
from mindspore.communication import init
from mindspore.common.initializer import initializer
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.parallel import Layout

ms.set_context(mode=ms.GRAPH_MODE)
rt.set_memory(max_size="28GB")
init()
ms.set_seed(1)

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = mint.flatten
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
        layout2 = Layout((8,), ("tp",))
        self.matmul1 = ms.parallel.shard(mint.matmul,
                                         in_strategy=(layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))
        self.relu1 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((4, 1),))
        self.matmul2 = ms.parallel.shard(mint.matmul, in_strategy=(layout2("None", "tp"), layout2("tp", "None")))
        self.relu2 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((8, 1),))
        self.matmul3 = mint.matmul

    def construct(self, x):
        x = self.flatten(x, start_dim=1, end_dim=3)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

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

def test_advanced_distributed_mint_operator_parallel():
    """
    Test advanced distributed mint operator parallel
    """
    data_set = create_dataset(32)

    net = Network()

    parallel_net = AutoParallel(net, parallel_mode="semi_auto")

    for epoch in range(10):
        i = 0
        for image, _ in data_set:
            forward_logits = parallel_net(image)
            if i % 10 == 0:
                forward_sum = mint.sum(forward_logits).asnumpy()
                print("epoch: %s, step: %s, forward_sum is %s" % (epoch, i, forward_sum))
            i += 1
