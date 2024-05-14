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
"""Transformation Infer Programming Guide"""

import os
import argparse
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.communication import init, get_rank
from mindspore.common.initializer import initializer

parser = argparse.ArgumentParser(description="Transform checkpoint dir")
parser.add_argument('--src_strategy_file',
                    type=str,
                    default="./src_strategy.ckpt",
                    help="The source strategy file.")
parser.add_argument("--dst_strategy_file",
                    type=str,
                    default="./dst_strategy.ckpt",
                    help="The destination strategy file.")
parser.add_argument("--src_checkpoints_dir",
                    type=str,
                    default="./src_checkpoints",
                    help="The source checkpoint directory.")
parser.add_argument("--dst_checkpoints_dir",
                    type=str,
                    default="./dst_checkpoints",
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
ms.set_auto_parallel_context(enable_parallel_optimizer=True)
init()
ms.set_seed(1)

class Dense(nn.Cell):
    """Dense layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = ms.Parameter(initializer("normal", [in_channels, out_channels], ms.float32))
        self.bias = ms.Parameter(initializer("normal", [out_channels], ms.float32))
        self.matmul = ops.MatMul()
        self.add = ops.Add()

    def construct(self, x):
        x = self.matmul(x, self.weight)
        x = self.add(x, self.bias)
        return x

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.layer1 = Dense(28*28, 512)
        self.relu1 = ops.ReLU()
        self.layer2 = Dense(512, 512)
        self.relu2 = ops.ReLU()
        self.layer3 = Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
net.layer1.matmul.shard(((1, 4), (4, 1)))
net.layer3.matmul.shard(((2, 2), (2, 1)))

predict_data = ms.Tensor(np.random.randn(1, 28, 28).astype(np.float32))
model = ms.Model(net)
model.infer_predict_layout(predict_data)
save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()),
                                    "checkpoint_{}.ckpt".format(get_rank()))
if bool(args_opt.only_compile):
    rank_list = ms.rank_list_for_transform(get_rank(), args_opt.src_strategy_file, args_opt.dst_strategy_file)
    checkpoint_file_map = {}
    for rank_id in rank_list:
        checkpoint_file_map[rank_id] = os.path.join(args_opt.src_checkpoints_dir,
                                                    "rank_{}".format(rank_id),
                                                    "checkpoint_{}.ckpt".format(rank_id))
    ms.transform_checkpoint_by_rank(get_rank(), checkpoint_file_map, save_checkpoint_path,
                                    args_opt.src_strategy_file, args_opt.dst_strategy_file)
else:
    param_dict = ms.load_checkpoint(save_checkpoint_path)
    ms.load_param_into_net(net, param_dict)
    predict_result = model.predict(predict_data)
    print(predict_result)
