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
"""
callback function
"""
import time
from mindspore import Callback
from mindspore import save_checkpoint

# stop training at runtime*60 second
class StopAtTime(Callback):
    """
    Args:
        run_time (float): set training time

    Example:
        >>> StopAtTime(1)
    """
    def __init__(self, run_time):
        super(StopAtTime, self).__init__()
        self.run_time = run_time*60

    def begin(self, run_context):
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        loss = cb_params.net_outputs
        cur_time = time.time()
        if (cur_time - cb_params.init_time) > self.run_time:
            print("epoch: ", epoch_num, " step: ", step_num, " loss: ", loss)
            run_context.request_stop()

    def end(self, run_context):
        cb_params = run_context.original_args()
        print(cb_params.list_callback)

class SaveCallback(Callback):
    """
    save the maximum accuracy checkpoint
    """
    def __init__(self, model, eval_dataset):
        super(SaveCallback, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.acc = 0.5

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        result = self.model.eval(self.eval_dataset)
        if result['accuracy'] > self.acc:
            self.acc = result['accuracy']
            file_name = str(self.acc) + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)
