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
"""MindIO TTP programming Guide"""
import os
import math
import random
import argparse
import numpy
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops, Parameter, train
from mindspore.communication import init, get_rank
from mindspore.common.initializer import initializer, HeUniform
from mindspore import get_ckpt_path_with_strategy, context

ms.set_seed(1)
numpy.random.seed(1)
random.seed(1)

parser = argparse.ArgumentParser(description="Mindio TTP test arguments")
parser.add_argument(
    "--is_recover",
    type=int,
    default=0,
    choices=[1, 0],
    help="Only used for resume from Mindio TTP checkpoint, default false.",
)
args_opt = parser.parse_args()

ms.set_context(mode=ms.GRAPH_MODE, jit_level="O1")
ms.set_device(device_target="Ascend")

ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
init()
ms.set_auto_parallel_context(
    strategy_ckpt_config={
        "save_file": "./src_pipeline_strategy/src_strategy_{}.ckpt".format(get_rank())
    }
)


class MatMulCell(nn.Cell):
    """
    MatMulCell definition.
    """

    def __init__(self, param=None, shape=None):
        super().__init__()
        if shape is None:
            shape = [28 * 28, 512]
        weight_init = HeUniform(math.sqrt(5))
        self.param = Parameter(initializer(weight_init, shape), name="param")
        if param is not None:
            self.param = param
        self.matmul = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        return out


class Network(nn.Cell):
    """
    Network definition.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = MatMulCell()
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 5120)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(5120, 5120)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Dense(5120, 512)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Dense(512, 10)

    def construct(self, x):
        """model construct"""
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.relu4(x)
        logits = self.layer5(x)
        return logits


net = Network()


def create_dataset(batch_size):
    """create dataset"""
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW(),
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, "image")
    dataset = dataset.map(label_transform, "label")
    dataset = dataset.batch(batch_size)
    return dataset


mini_dataset = create_dataset(32)

optimizer = nn.SGD(net.trainable_params(), 1e-2)
# 配置TFT优化器
optimizer_wrapper = nn.OptTFTWrapper(optimizer)

# 配置loss函数
loss_fn = nn.CrossEntropyLoss()
net.set_train()
init_epoch = 0

if bool(args_opt.is_recover):
    cur_epoch = 2  # 设置成异常保存的epoch值
    cur_step = 1215  # 设置成异常保存的step值
    ckpt_step = (cur_epoch - 1) * mini_dataset.get_dataset_size() + cur_step
    if context.get_auto_parallel_context("parallel_mode") == "data_parallel":
        cur_rank = 0
        file_name = f"ttp_rank_{cur_rank}-{cur_epoch}_{cur_step}.ckpt"
        new_ckpt_file = f"./ttp_checkpoints/tft_saved_checkpoints-step_{ckpt_step}/rank_{cur_rank}/{file_name}"
    else:
        cur_rank = get_rank()
        file_name = f"ttp_rank_{cur_rank}-{cur_epoch}_{cur_step}.ckpt"
        ckpt_file = f"./ttp_checkpoints/tft_saved_checkpoints-step_{ckpt_step}/rank_{cur_rank}/{file_name}"
        strategy_file = f"./src_pipeline_strategy/src_strategy_{cur_rank}.ckpt"
        new_ckpt_file = get_ckpt_path_with_strategy(ckpt_file, strategy_file)
    param_dict = ms.load_checkpoint(new_ckpt_file)
    ms.load_param_into_net(net, param_dict)
    mini_dataset.set_init_step(int(param_dict["step_num"]))
    init_epoch = int(param_dict["epoch_num"]) - 1

# 配置model对象
model = ms.Model(net, loss_fn, optimizer=optimizer_wrapper)

time_monitor = train.TimeMonitor(data_size=1)
loss_cb = train.LossMonitor(1)

# 设置TFT callback对象
tft_cb = train.TrainFaultTolerance()

model.train(
    5, mini_dataset, callbacks=[time_monitor, loss_cb, tft_cb], initial_epoch=init_epoch
)
