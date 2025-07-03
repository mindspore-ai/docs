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
"""For loop Profiler Example"""
import numpy as np

import mindspore
import mindspore.dataset as ds
from mindspore import nn
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator_net():
    for _ in range(5):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits


def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss


if __name__ == "__main__":
    mindspore.set_device("Ascend")
    model = Net()
    optimizer = nn.Momentum(model.trainable_params(), 1, 0.9)
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    all_data = ds.GeneratorDataset(generator_net(), ["data", "label"])

    # Init Profiler
    # pylint: disable=protected-access
    experimental_config = mindspore.profiler._ExperimentalConfig(
        profiler_level=ProfilerLevel.Level0,
        aic_metrics=AicoreMetrics.AiCoreNone,
        l2_cache=False,
        mstx=False,
        data_simplification=False,
    )
    # Note that the Profiler should be initialized before model.train
    with mindspore.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=mindspore.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1, skip_first=0
            ),
            on_trace_ready=mindspore.profiler.tensorboard_trace_handler("./data"),
            profile_memory=False,
            experimental_config=experimental_config,
    ) as prof:
        # Train Model
        for step_data, step_label in all_data:
            train_step(step_data, step_label)
            prof.step()
