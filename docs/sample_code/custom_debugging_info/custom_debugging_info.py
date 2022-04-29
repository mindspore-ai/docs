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

"""Custom Debugging Info Tutorial
This sample code is applicable to CPU, GPU and Ascend.
"""
import os
import json
from mindspore import log as logger, set_context, GRAPH_MODE
from mindspore import Model
import mindspore.nn as nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from src.lenet import LeNet5
from src.datasets import create_dataset
from custom_callback import StopAtTime


def set_dump_info():
    """
    set the dump parameter and write it in the JSON file of this directory
    """
    abspath = os.getcwd()
    data_dump = {
        "common_dump_settings": {
            "dump_mode": 0,
            "path": abspath + "/data_dump",
            "net_name": "LeNet5",
            "iteration": "0|5-8|100-120",
            "input_output": 2,
            "kernels": ["Default/network-WithLossCell/_backbone-LeNet5/flatten-Flatten/Reshape-op118"],
            "support_device": [0, 1, 2, 3, 4, 5, 6, 7]
        },
        "e2e_dump_settings": {
            "enable": True,
            "trans_flag": False
        }
    }
    with open("./data_dump.json", "w", encoding="GBK") as f:
        json.dump(data_dump, f)
    os.environ['MINDSPORE_DUMP_CONFIG'] = abspath + "/data_dump.json"

def set_log_info():
    os.environ['GLOG_v'] = '1'
    os.environ['GLOG_logtostderr'] = '1'
    os.environ['logger_maxBytes'] = '5242880'
    os.environ['GLOG_log_dir'] = 'D:/' if os.name == "nt" else '/var/log/mindspore'
    os.environ['logger_backupCount'] = '10'
    print(logger.get_log_config())

if __name__ == "__main__":
    # clean files
    if os.name == "nt":
        os.system('del/f/s/q *.ckpt *.meta')
    else:
        os.system('rm -f *.ckpt *.meta *.pb')

    set_dump_info()
    set_log_info()

    set_context(mode=GRAPH_MODE, device_target="CPU")
    lr = 0.01
    momentum = 0.9
    epoch_size = 3
    train_data_path = "./datasets/MNIST_Data/train"
    eval_data_path = "./datasets/MNIST_Data/test"
    model_path = "./models/ckpt/custom_debugging_info/"

    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    repeat_size = 1
    network = LeNet5()

    metrics = {
        'accuracy': nn.Accuracy(),
        'loss': nn.Loss(),
        'precision': nn.Precision(),
        'recall': nn.Recall(),
        'f1_score': nn.F1()
        }
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=model_path, config=config_ck)

    model = Model(network, net_loss, net_opt, metrics=metrics)

    print("============== Starting Training ==============")
    ds_train = create_dataset(train_data_path, repeat_size=repeat_size)
    stop_cb = StopAtTime(run_time=0.6)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(375), stop_cb], dataset_sink_mode=False)

    print("============== Starting Testing ==============")
    ds_eval = create_dataset(eval_data_path, repeat_size=repeat_size)
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))
