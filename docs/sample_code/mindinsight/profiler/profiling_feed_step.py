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
"""Start profiling on step in custom training."""
import numpy as np
from mindspore import nn
import mindspore as ms
import mindspore.dataset as ds


class StopAtStep(ms.Callback):
    """
    Start profiling base on step.

    Args:
        start_step (int): The start step number.
        stop_step (int): The stop step number.
    """
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = ms.Profiler(start_profile=False, output_path='./data_step')

    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()

    def end(self, run_context):
        self.profiler.analyse()


class Net(nn.Cell):
    """The test net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator():
    for _ in range(10):
        yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))


if __name__ == '__main__':
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

    profile_call_back = StopAtStep(5, 8)

    net = Net()
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator, ["data", "label"])
    model = ms.Model(net, loss, optimizer)
    model.train(3, data, callbacks=[profile_call_back], dataset_sink_mode=False)
