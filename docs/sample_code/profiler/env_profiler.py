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
"""Env Profiler Example"""
import os
import numpy as np

# Set the environment variable for the Profiler before import mindspore
os.environ['MS_PROFILER_OPTIONS'] = (
    '{"start": true, "output_path": "./data", "activities": ["CPU", "NPU"], "with_stack": true, '
    '"aicore_metrics": "AicoreNone", "l2_cache": false, "profiler_level": "Level0"}')

# pylint: disable=wrong-import-position
import mindspore
import mindspore.dataset as ds
from mindspore import nn


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator_net():
    for _ in range(2):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits


def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss


if __name__ == '__main__':
    model = Net()
    optimizer = nn.Momentum(model.trainable_params(), 1, 0.9)
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    all_data = ds.GeneratorDataset(generator_net(), ["data", "label"])

    for step_data, step_label in all_data:
        train_step(step_data, step_label)
