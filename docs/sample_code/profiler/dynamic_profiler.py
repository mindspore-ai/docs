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
"""Dynamic profiler Example"""
import json
import os
import numpy as np

import mindspore
import mindspore.dataset as ds
from mindspore import nn
from mindspore.profiler import DynamicProfilerMonitor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator_net():
    for _ in range(15):
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
    mindspore.set_device("Ascend")
    model = Net()
    optimizer = nn.Momentum(model.trainable_params(), 1, 0.9)
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    all_data = ds.GeneratorDataset(generator_net(), ["data", "label"])

    # set json configuration file
    data_cfg = {
        "start_step": 2,
        "stop_step": 5,
        "aic_metrics": "AiCoreNone",
        "profiler_level": "Level0",
        "analyse_mode": 0,
        "activities": ["CPU", "NPU"],
        "export_type": ["text"],
        "profile_memory": False,
        "mstx": False,
        "parallel_strategy": False,
        "with_stack": False,
        "data_simplification": True,
        "l2_cache": False,
        "analyse": True,
        "record_shape": False,
        "prof_path": "./data",
        "mstx_domain_include": [],
        "mstx_domain_exclude": [],
        "host_sys": [],
        "sys_io": False,
        "sys_interconnection": False
    }
    output_path = "./cfg_path"
    cfg_path = os.path.join(output_path, "profiler_config.json")
    os.makedirs(output_path, exist_ok=True)
    # set cfg file
    with open(cfg_path, 'w') as f:
        json.dump(data_cfg, f, indent=4)

    # Define a network of training models
    dp = DynamicProfilerMonitor(cfg_path=output_path)
    for step_data, step_label in all_data:
        train_step(step_data, step_label)
        # Call step collection
        dp.step()
