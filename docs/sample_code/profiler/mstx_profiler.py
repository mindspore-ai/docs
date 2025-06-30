# Copyright 2025 Huawei Technologies Co., Ltd
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
"""mstx Profiler Example"""
import numpy as np

import mindspore
import mindspore.dataset as ds
from mindspore import nn
from mindspore.profiler import ProfilerLevel, ProfilerActivity, schedule, tensorboard_trace_handler, mstx


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator_net():
    for _ in range(10):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


def forward_fn(data, label):
    logits = model(data)
    mstx.mark("backward_begin")
    loss = loss_fn(logits, label)
    return loss, logits


def train_step(data, label):
    range_id1 = mstx.range_start("forward_and_backward")
    (loss, _), grads = grad_fn(data, label)
    mstx.range_end(range_id1)
    range_id2 = mstx.range_start("optimizer_step")
    optimizer(grads)
    mstx.range_end(range_id2)
    return loss


if __name__ == "__main__":
    mindspore.set_device("Ascend")
    model = Net()
    optimizer = nn.Momentum(model.trainable_params(), 1, 0.9)
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    stream = mindspore.runtime.current_stream()
    # pylint: disable=protected-access
    experimental_config = mindspore.profiler._ExperimentalConfig(
        profiler_level=ProfilerLevel.LevelNone,
        mstx=True)
    with mindspore.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler("./data"),
            experimental_config=experimental_config
    ) as profiler:
        for step_data, step_label in ds.GeneratorDataset(generator_net(), ["data", "label"]):
            train_step(step_data, step_data)
            profiler.step()
