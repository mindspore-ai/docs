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
"""Transformation Retrain Programming Guide"""

import os
import argparse
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, train
from mindspore.communication import init, get_rank

parser = argparse.ArgumentParser(description="Transform checkpoint dir")
parser.add_argument('--src_strategy_dir',
                    type=str,
                    default="./src_pipeline_strategys",
                    help="The source strategy file dir.")
parser.add_argument('--src_strategy_file',
                    type=str,
                    default="./src_pipeline_strategy.ckpt",
                    help="The source strategy file.")
parser.add_argument("--dst_strategy_file",
                    type=str,
                    default="./dst_pipeline_strategy.ckpt",
                    help="The destination strategy file.")
parser.add_argument("--src_checkpoints_dir",
                    type=str,
                    default="./src_checkpoints_pipeline",
                    help="The source checkpoint directory.")
parser.add_argument("--dst_checkpoints_dir",
                    type=str,
                    default="./dst_checkpoints_pipeline",
                    help="The destination checkpoint directory.")
parser.add_argument("--only_compile",
                    type=int,
                    default=0,
                    choices=[1, 0],
                    help="Only compile and convert the net, default is disable.")
args_opt = parser.parse_args()

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(strategy_ckpt_config={"save_file": args_opt.dst_strategy_file})
ms.set_auto_parallel_context(full_batch=True)
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

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)

if bool(args_opt.only_compile):
    model.infer_train_layout(data_set)
    if get_rank() == 0:
        ms.merge_pipeline_strategys(args_opt.src_strategy_dir, args_opt.src_strategy_file)
        ms.transform_checkpoints(args_opt.src_checkpoints_dir, args_opt.dst_checkpoints_dir,
                                 "checkpoint_", args_opt.src_strategy_file, args_opt.dst_strategy_file)
else:
    save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()),
                                        "checkpoint_{}.ckpt".format(get_rank()))
    loss_cb = train.LossMonitor(20)
    model.infer_train_layout(data_set)
    param_dict = ms.load_checkpoint(save_checkpoint_path)
    ms.load_param_into_net(net, param_dict)
    model.train(2, data_set, callbacks=[loss_cb])
