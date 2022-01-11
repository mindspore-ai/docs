# Copyright 2021 Huawei Technologies Co., Ltd
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
"""train
This sample code is applicable to GPU and Ascend.
"""
import argparse
import os

import mindspore.nn as nn
from mindspore import Model, context
from mindspore.nn import TrainOneStepCell, Accuracy
from mindspore.boost import GradientAccumulation
import mindspore.ops as ops
from mindspore.train.callback import LossMonitor, TimeMonitor

from models.official.cv.lenet.src.dataset import create_dataset
from models.official.cv.lenet.src.lenet import LeNet5

class TrainGradAccumulationStepsCell(TrainOneStepCell):
    """construct train accu step cell"""
    def __init__(self, network, optimizer, sens=1.0, max_accumulation_step=1):
        super(TrainGradAccumulationStepsCell, self).__init__(network, optimizer, sens)
        self.max_accumulation_step = max_accumulation_step
        self.grad_accumulation = GradientAccumulation(self.max_accumulation_step, self.optimizer)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = ops.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = self.grad_accumulation(loss, grads)
        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Grad Cumulative Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./Data",
                        help='path where the dataset is saved')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    ds_train = create_dataset(os.path.join(args.data_path, "train"), 32)

    net = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())

    train_net = nn.WithLossCell(net, net_loss)
    train_net = TrainGradAccumulationStepsCell(train_net, net_opt, 1.0, 5)
    model = Model(train_net)

    print("============== Starting Training ==============")
    model.train(10, ds_train, callbacks=[time_cb, LossMonitor()])

    print("============== Starting Testing ==============")
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    ds_eval = create_dataset(os.path.join(args.data_path, "test"), 32, 1)
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
