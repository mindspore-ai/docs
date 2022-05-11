# Copyright 2020 Huawei Technologies Co., Ltd
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
"""ResNet50 model train with MindSpore
This sample code is applicable to GPU and Ascend
"""
import os
import argparse
import random
import time
import numpy as np
import moxing as mox

from mindspore import Tensor, Model, ParallelMode, set_context, GRAPH_MODE, set_auto_parallel_context
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore import Callback, LossMonitor
from mindspore import FixedLossScaleManager
from mindspore.communication import init
import mindspore.dataset as ds

from dataset import create_dataset, device_id, device_num
from resnet import resnet50

random.seed(1)
np.random.seed(1)
ds.config.set_seed(1)


class PerformanceCallback(Callback):
    """
    Training performance callback.

    Args:
        batch_size (int): Batch number for one step.
    """
    def __init__(self, batch_size):
        super(PerformanceCallback, self).__init__()
        self.batch_size = batch_size
        self.last_step = 0
        self.epoch_begin_time = 0

    def step_begin(self, run_context):
        self.epoch_begin_time = time.time()

    def step_end(self, run_context):
        params = run_context.original_args()
        cost_time = time.time() - self.epoch_begin_time
        train_steps = params.cur_step_num -self.last_step
        print(f'epoch {params.cur_epoch_num} cost time = {cost_time}, train step num: {train_steps}, '
              f'one step time: {1000*cost_time/train_steps} ms, '
              f'train samples per second of cluster: {device_num*train_steps*self.batch_size/cost_time:.1f}\n')
        self.last_step = run_context.original_args().cur_step_num


def get_lr(global_step,
           total_epochs,
           steps_per_epoch,
           lr_init=0.01,
           lr_max=0.1,
           warmup_epochs=5):
    """
    Generate learning rate array.

    Args:
        global_step (int): Initial step of training.
        total_epochs (int): Total epoch of training.
        steps_per_epoch (float): Steps of one epoch.
        lr_init (float): Initial learning rate. Default: 0.01.
        lr_max (float): Maximum learning rate. Default: 0.1.
        warmup_epochs (int): The number of warming up epochs. Default: 5.

    Returns:
        np.array, learning rate array.
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(int(total_steps)):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr = float(lr_max) * base * base
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def resnet50_train(args):
    """Training the ResNet-50."""
    epoch_size = args.epoch_size
    batch_size = 32
    class_num = 10
    loss_scale_num = 1024
    local_data_path = '/cache/data'

    # set graph mode and parallel mode
    set_context(mode=GRAPH_MODE, device_target="Ascend", save_graphs=False)
    set_context(device_id=device_id)
    if device_num > 1:
        set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        init()
        local_data_path = os.path.join(local_data_path, str(device_id))

    # data download
    print('Download data.')
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_path)

    # create dataset
    print('Create train and evaluate dataset.')
    train_dataset = create_dataset(dataset_path=local_data_path, do_train=True,
                                   repeat_num=1, batch_size=batch_size)
    eval_dataset = create_dataset(dataset_path=local_data_path, do_train=False,
                                  repeat_num=1, batch_size=batch_size)
    train_step_size = train_dataset.get_dataset_size()
    print('Create dataset success.')

    # create model
    net = resnet50(class_num=class_num)
    # reduction='mean' means that apply reduction of mean to loss
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    lr = Tensor(get_lr(global_step=0, total_epochs=epoch_size, steps_per_epoch=train_step_size))
    opt = Momentum(net.trainable_params(), lr, momentum=0.9, weight_decay=1e-4, loss_scale=loss_scale_num)
    loss_scale = FixedLossScaleManager(loss_scale_num, False)

    # amp_level="O2" means that the hybrid precision of O2 mode is used for training
    # the whole network except that batchnoram will be cast into float16 format and dynamic loss scale will be used
    # 'keep_batchnorm_fp32 = False' means that use the float16 format
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'}, amp_level="O2", keep_batchnorm_fp32=False,
                  loss_scale_manager=loss_scale)

    # define performance callback to show ips and loss callback to show loss for every epoch
    performance_cb = PerformanceCallback(batch_size)
    loss_cb = LossMonitor()
    cb = [performance_cb, loss_cb]

    print(f'Start run training, total epoch: {epoch_size}.')
    model.train(epoch_size, train_dataset, callbacks=cb)
    if device_num == 1 or device_id == 0:
        print(f'Start run evaluation.')
        output = model.eval(eval_dataset)
        print(f'Evaluation result: {output}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet50 train.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--epoch_size', type=int, default=90, help='Train epoch size.')

    args_opt, unknown = parser.parse_known_args()

    resnet50_train(args_opt)
    print('ResNet50 training success!')
