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
"""Pipeline Train Programming Guide"""

import os
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, train
from mindspore.communication import init, get_rank

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(pipeline_stages=2, enable_parallel_optimizer=True)
init()
ms.set_auto_parallel_context(strategy_ckpt_config={
    "save_file": "./src_pipeline_strategys/src_strategy_{}.ckpt".format(get_rank())})
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
net.layer2.pipeline_stage = 1
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
loss_cb = train.LossMonitor(20)
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=1, integrated_save=False)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint",
                                   directory="./src_checkpoints_pipeline/rank_{}".format(get_rank()),
                                   config=ckpt_config)
net_with_grads = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
model = ms.Model(net_with_grads, optimizer=optimizer)
model.train(3, data_set, callbacks=[loss_cb, ckpoint_cb])
