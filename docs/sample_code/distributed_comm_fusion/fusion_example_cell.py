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
"""Cell Fusion Example"""
import os
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()

class DenseLayer(nn.Cell):
    """A base layer with two dense layer"""
    def __init__(self):
        super().__init__()
        self.input_mapping = nn.Dense(10, 32)
        self.output_mapping = nn.Dense(32, 10)

    def construct(self, x):
        x = self.input_mapping(x)
        return self.output_mapping(x)

class Net(nn.Cell):
    """An network with many dense layers"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.head = nn.Dense(28*28, 10)
        self.layer1 = DenseLayer()
        self.layer2 = DenseLayer()
        self.layer3 = DenseLayer()

    def construct(self, x):
        x = self.flatten(x)
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

net = Net()
# 配置通信融合
net.head.set_comm_fusion(0)
net.layer1.set_comm_fusion(1)
net.layer2.set_comm_fusion(2)
net.layer3.set_comm_fusion(3)
for item in net.trainable_params():
    print(f"The parameter {item.name}'s fusion id is {item.comm_fusion}")

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

def forward_fn(data, target):
    """forward propagation"""
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    """train_step"""
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = train_step(image, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
