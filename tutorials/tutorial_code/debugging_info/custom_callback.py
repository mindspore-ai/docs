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
The sample can be run on CPU/GPU/Ascend.
"""
import time
import mindspore.nn as nn
from mindspore.nn import Momentum
from mindspore import Model, context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback import Callback, LossMonitor

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
        result = self.model.eval(self.ds_eval)
        if result['Accuracy'] > self.acc:
            self.acc = result['Accuracy']
            file_name = str(self.acc) + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint, the accuracy is", self.acc)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE)
    train_dataset = create_train_dataset()
    eval_dataset = create_eval_dataset()
    net = Net()
    net_opt = Momentum(net.trainable_params(), 0.01, 0.9)
    net_loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
    model = Model(network=net, loss_fn=net_loss, optimizer=net_opt, metrics={'Accuracy': nn.Accuracy()})
    model.train(epoch=100,
                train_dataset=train_dataset,
                callbacks=[LossMonitor(), StopAtTime(3), SaveCallback(model, eval_dataset)])
