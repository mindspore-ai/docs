# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""train resnet."""
import os
import argparse
import ast
from mindspore import set_seed, Model, ParallelMode, set_context, GRAPH_MODE, set_auto_parallel_context
from mindspore.nn import Momentum
from mindspore import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication import init
from mindspore.common import initializer
import mindspore.nn as nn

from src.config import config
from src.dataset import create_dataset
from src.resnet import resnet50
from src.cross_entropy_smooth import CrossEntropySmooth

set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt = parser.parse_args()


if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_size = int(os.getenv('RANK_SIZE', '1'))
    rank_id = int(os.getenv('RANK_ID', '0'))

    # init context
    set_context(mode=GRAPH_MODE, device_target='Ascend', device_id=device_id)
    if rank_size > 1:
        set_auto_parallel_context(device_num=rank_size, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        init()

    # create dataset
    dataset = create_dataset(args_opt.dataset_path, config.batch_size, rank_size, rank_id)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet50(class_num=config.class_num)

    # init weight
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer.initializer(initializer.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer.initializer(initializer.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))

    lr = nn.dynamic_lr.cosine_decay_lr(config.lr_end, config.lr, config.epoch_size * step_size,
                                       step_size, config.warmup)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    opt = Momentum(group_params, lr, config.momentum)
    # define loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=config.label_smooth_factor,
                              num_classes=config.class_num)
    # define model
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        #config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
        config_ck = CheckpointConfig(save_checkpoint_steps=5,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=config.save_checkpoint_path, config=config_ck)
        cb += [ckpt_cb]

    model.train(config.epoch_size, dataset, callbacks=cb, sink_size=step_size, dataset_sink_mode=False)
