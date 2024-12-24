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

"""KBK Profiler step Example"""

import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context, nn, Profiler
from mindspore.profiler import schedule, tensor_board_trace_handler


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator_net():
    for _ in range(2):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


def train(test_net):
    optimizer = nn.Momentum(test_net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator_net(), ["data", "label"])
    model = ms.train.Model(test_net, loss, optimizer)
    model.train(1, data)


if __name__ == '__main__':
    context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

    net = Net()
    STEP_NUM = 15

    # with mode
    # Note that the result of skip_first + (wait + warm_up + active) * repeat should not be greater than STEP_NUM
    with Profiler(schedule=schedule(wait=1, warm_up=1, active=2, repeat=1, skip_first=2),
                  on_trace_ready=tensor_board_trace_handler) as prof:
        for _ in range(STEP_NUM):
            train(net)
            prof.step()
