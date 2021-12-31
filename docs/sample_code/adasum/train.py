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
"""train resnet."""
import argparse
import os

from mindspore import context
from mindspore import Tensor
from mindspore.nn import Momentum
from mindspore import Model
from mindspore import ParallelMode
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore import FixedLossScaleManager
from mindspore.communication import init
from mindspore import set_seed
from mindspore.parallel import set_algo_parameters
from mindspore import nn
from mindspore.common import initializer as weight_init

from models.official.cv.resnet.src.lr_generator import get_lr
from models.official.cv.resnet.src.CrossEntropySmooth import CrossEntropySmooth
from models.official.cv.resnet.src.dataset import create_dataset2 as create_dataset
from models.official.cv.resnet.src.metric import ClassifyCorrectCell

from resnet import resnet50 as resnet


set_seed(1)


def init_weight(network):
    """init_weight"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))


def init_group_params(network):
    """init_group_params"""
    decayed_params = []
    no_decayed_params = []
    for param in network.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_parameters = [{'params': decayed_params, 'weight_decay': 0.0001},
                        {'params': no_decayed_params},
                        {'order_params': network.trainable_params()}]
    return group_parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore Dimension Reduce Training Example Stage 1')
    parser.add_argument('--data_path', type=str, default="./data", help='path where the dataset is saved')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)
    context.set_auto_parallel_context(device_num=16, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    set_algo_parameters(elementwise_op_strategy_follow=True)
    init()

    # define train dataset
    train_data_path = os.path.join(args.data_path, "train")
    ds_train = create_dataset(dataset_path=train_data_path, do_train=True, batch_size=256, train_image_size=224,
                              eval_image_size=224, target="Ascend", distribute=True)
    step_size = ds_train.get_dataset_size()

    # define net
    net = resnet(num_classes=1001)
    init_weight(network=net)

    # define loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, num_classes=1001)
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)

    # define optimizer
    group_params = init_group_params(net)
    lr = get_lr(lr_init=0, lr_end=0.0, lr_max=0.8, warmup_epochs=5, total_epochs=90, steps_per_epoch=step_size,
                lr_decay_mode="linear")
    lr = Tensor(lr)
    opt = Momentum(group_params, lr, 0.9, loss_scale=1024)

    # define eval_network
    dist_eval_network = ClassifyCorrectCell(net)

    # define boost config dictionary
    boost_dict = {
        "boost": {
            "mode": "manual",
            "less_bn": False,
            "grad_freeze": False,
            "adasum": True,
            "grad_accumulation": False,
            "dim_reduce": False
        }
    }

    # define model
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, amp_level="O2", boost_level="O2",
                  keep_batchnorm_fp32=False, boost_config_dict=boost_dict, eval_network=dist_eval_network)

    # define callback
    cb = [TimeMonitor(data_size=step_size), LossMonitor()]

    print("============== Starting Training ==============")
    model.train(90, ds_train, callbacks=cb, sink_size=step_size, dataset_sink_mode=True)
