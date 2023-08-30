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
"""Manual Parallel Programming Guide"""

import os
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops, train
from mindspore.communication import init, get_rank, get_group_size

init()
ms.set_seed(1)

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = x[get_rank():get_rank()+32//get_group_size()]
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()

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

class ReduceLoss(nn.Cell):
    """create loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.all_reduce = ops.AllReduce()

    def construct(self, data, label):
        label = label[get_rank():get_rank()+32//get_group_size()]
        loss_value = self.loss(data, label)
        loss_value = self.all_reduce(loss_value) / get_group_size()
        return loss_value

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = ReduceLoss()
loss_cb = train.LossMonitor(20)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb])
