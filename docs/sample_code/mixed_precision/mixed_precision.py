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

"""Mixed Precision Tutorial
This sample code is applicable to GPU and Ascend.
"""
import mindspore.nn as nn
from mindspore import Model, set_context, GRAPH_MODE
from mindspore import LossMonitor
from mindspore.nn import Accuracy
from src.lenet import LeNet5
from src.datasets import create_dataset


if __name__ == "__main__":

    set_context(mode=GRAPH_MODE, device_target="GPU")

    ds_train = create_dataset("./datasets/MNIST_Data/train", 32)
    ds_eval = create_dataset("./datasets/MNIST_Data/test", 32)
    # Initialize network
    network = LeNet5(10)

    # Define Loss and Optimizer
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
    # amp_leval=O2 in GPU, amp_leval=O3 in Ascend, O0 is without mixed precision
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O2")

    # Run training
    model.train(epoch=1, callbacks=[LossMonitor()], train_dataset=ds_train)

    # Run training
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("====Accuracy====:", acc)
