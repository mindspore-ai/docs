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
"""custom callback
This sample code is applicable to Ascend, CPU and GPU.
"""
import os
import time
import json
import mindspore.nn as nn
from mindspore.nn import Momentum, SoftmaxCrossEntropyWithLogits
from mindspore import Model, save_checkpoint, set_context, GRAPH_MODE
from mindspore.train.callback import Callback, LossMonitor
from mindspore import log as logger

from src.dataset import create_train_dataset, create_eval_dataset
from src.net import Net


class StopAtTime(Callback):
    """StopAtTime"""
    def __init__(self, run_time):
        """init"""
        super(StopAtTime, self).__init__()
        self.run_time = run_time*60

    def begin(self, run_context):
        """begin"""
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        loss = cb_params.net_outputs
        cur_time = time.time()
        if (cur_time - cb_params.init_time) > self.run_time:
            print(f"Stop after {self.run_time}s.")
            print(f"epoch: {epoch_num}, step: {step_num}, loss is {loss}")
            run_context.request_stop()


class SaveCallback(Callback):
    """SaveCallback"""
    def __init__(self, eval_model, ds_eval):
        """init"""
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.acc = 0

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        result = self.model.eval(self.ds_eval, dataset_sink_mode=False)
        if result['Accuracy'] > self.acc:
            self.acc = result['Accuracy']
            file_name = str(self.acc) + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint, the accuracy is", self.acc)

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
    set_dump_info()
    set_log_info()
    set_context(mode=GRAPH_MODE)
    train_dataset = create_train_dataset()
    eval_dataset = create_eval_dataset()
    net = Net()
    net_opt = Momentum(net.trainable_params(), 0.01, 0.9)
    net_loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
    model = Model(network=net, loss_fn=net_loss, optimizer=net_opt, metrics={'Accuracy': nn.Accuracy()})
    model.train(epoch=100,
                train_dataset=train_dataset,
                callbacks=[LossMonitor(), StopAtTime(3), SaveCallback(model, eval_dataset)], dataset_sink_mode=False)
