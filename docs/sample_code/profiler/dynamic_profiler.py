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
from mindspore import context, nn
from mindspore.profiler import DynamicProfilerMonitor


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
    model = mindspore.train.Model(test_net, loss, optimizer)
    model.train(1, data)


def change_cfg_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data['start_step'] = 6
    data['stop_step'] = 7

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # set json configuration file
    cfg_json = {
        "start_step": 2,
        "stop_step": 5,
        "aicore_metrics": 1,
        "profiler_level": -1,
        "profile_framework": 1,
        "analyse_mode": 0,
        "with_stack": True,
        "parallel_strategy": True,
        "data_simplification": False,
    }
    context.set_context(mode=mindspore.PYNATIVE_MODE)
    mindspore.set_device("Ascend")

    cfg_path = os.path.join("./cfg_path", "profiler_config.json")
    # set cfg file
    with open(cfg_path, 'w') as f:
        json.dump(cfg_json, f, indent=4)

    # Assume the user has correctly configured the environment variable (RANK_ID is not a non-numeric type)
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0

    # cfg_path contains the json configuration file path, and output_path is the output path
    dp = DynamicProfilerMonitor(cfg_path=cfg_path, output_path=cfg_path)
    STEP_NUM = 15
    # Define a network of training models
    net = Net()
    for i in range(STEP_NUM):
        train(net)
        # Modify the configuration file after step 7. For example, change start_step to 8 and stop_step to 10
        if i == 7:
            # Modify parameters in the JSON file
            change_cfg_json(os.path.join(cfg_path, "profiler_config.json"))
        # Call step collection
        dp.step()
