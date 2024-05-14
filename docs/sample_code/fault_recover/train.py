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

"""Fault Recover Example"""

import os
import argparse
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops, train
from mindspore.communication import init, get_rank
from mindspore.common.initializer import initializer

parser = argparse.ArgumentParser(description="Transform checkpoint dir")
parser.add_argument("--is_recover",
                    type=int,
                    default=0,
                    choices=[1, 0],
                    help="Only compile and convert the net, default is disable.")
args_opt = parser.parse_args()

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(enable_parallel_optimizer=True)
init()
ms.set_seed(1)

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul()
        self.relu1 = ops.ReLU()
        self.matmul2 = ops.MatMul()
        self.relu2 = ops.ReLU()
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

net = Network()
net.matmul1.shard(((2, 4), (4, 1)))
net.relu1.shard(((4, 1),))

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
loss_cb = train.LossMonitor()
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=4, integrated_save=False)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint",
                                   directory="./checkpoints/rank_{}".format(get_rank()),
                                   config=ckpt_config)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)

if bool(args_opt.is_recover):
    param_dict = ms.load_checkpoint("./checkpoints/rank_{}/checkpoint-2_1875.ckpt".format(get_rank()))
    model.infer_train_layout(data_set)
    ms.load_param_into_net(net, param_dict)

model.train(2, data_set, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)
