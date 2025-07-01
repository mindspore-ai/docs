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
"""Call back start stop example."""
import numpy as np

from mindspore import nn
import mindspore
import mindspore.dataset as ds


class StopAtStep(mindspore.Callback):
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
        # pylint: disable=protected-access
        experimental_config = mindspore.profiler._ExperimentalConfig()
        self.profiler = mindspore.profiler.profile(start_profile=False, experimental_config=experimental_config,
                                                   schedule=mindspore.profiler.schedule(wait=0, warmup=0,
                                                                                        active=self.stop_step -
                                                                                        self.start_step + 1,
                                                                                        repeat=1, skip_first=0),
                                                   on_trace_ready=mindspore.profiler.tensorboard_trace_handler(
                                                       "./data"))

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if self.start_step <= step_num <= self.stop_step:
            self.profiler.step()
        if step_num == self.stop_step:
            self.profiler.stop()


class Net(nn.Cell):
    """The test net"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator():
    for _ in range(10):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


if __name__ == '__main__':
    mindspore.set_device("Ascend")

    profile_call_back = StopAtStep(5, 8)

    net = Net()
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator, ["data", "label"])
    model = mindspore.Model(net, loss, optimizer)
    model.train(3, data, callbacks=[profile_call_back], dataset_sink_mode=False)
