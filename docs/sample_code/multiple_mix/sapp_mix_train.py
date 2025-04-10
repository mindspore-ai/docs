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
"""Sharding Propagation Mix Programming Guide"""

import os
import mindspore as ms
import mindspore.dataset as ds
import mindspore.runtime as rt
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.nn.utils import no_init_parameters
from mindspore import nn, train
from mindspore.communication import init
class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 1)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

def create_dataset(batch_size):
    """create dataset"""
    dataset_path = "./MNIST_Data/train"
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

def test_parallel_sapp():
    """set stage_config and train"""
    os.environ['MS_DEV_SAVE_GRAPHS'] = '2'
    ms.set_context(mode=ms.GRAPH_MODE)
    rt.set_memory(max_size="25GB")
    init()
    ms.set_seed(1)
    with no_init_parameters():
        net = Network()
        optimizer = nn.SGD(net.trainable_params(), 1e-2)
    # 配置relu算子的重计算
    net.relu1.recompute()
    net.relu2.recompute()
    data_set = create_dataset(32)
    loss_fn = nn.MAELoss()
    loss_cb = train.LossMonitor()
    # 配置每一层在流水线并行中的pipeline_stage编号
    net_with_grads = ms.parallel.nn.Pipeline(nn.WithLossCell(net, loss_fn), 4,
                                             stage_config={"_backbone.layer1": 0,
                                                           "_backbone.relu1": 0,
                                                           "_backbone.layer2": 1,
                                                           "_backbone.relu2": 1,
                                                           "_backbone.layer3": 1,})
    net_with_grads_new = AutoParallel(net_with_grads, parallel_mode="recursive_programming")
    net_with_grads_new.full_batch = True
    net_with_grads_new.pipeline(stages=2, scheduler="1f1b")
    model = ms.Model(net_with_grads_new, optimizer=optimizer)
    model.train(10, data_set, callbacks=[loss_cb], dataset_sink_mode=True)
