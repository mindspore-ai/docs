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

import mindspore as ms
from mindspore.nn import Momentum
from mindspore.communication import init
from mindspore import nn
from mindspore.common import initializer as weight_init

from models.official.cv.resnet.src.lr_generator import get_lr
from models.official.cv.resnet.src.CrossEntropySmooth import CrossEntropySmooth
from models.official.cv.resnet.src.dataset import create_dataset2 as create_dataset
from models.official.cv.resnet.src.model_utils.local_adapter import get_rank_id

from resnet import resnet50 as resnet


ms.set_seed(1)


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

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    device_id = int(os.getenv('DEVICE_ID'))
    ms.set_context(device_id=device_id)
    ms.set_auto_parallel_context(device_num=8, parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    all_reduce_fusion_config = [85, 160]
    ms.set_auto_parallel_context(all_reduce_fusion_config=all_reduce_fusion_config)
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
    loss_scale = ms.FixedLossScaleManager(1024, drop_overflow_update=False)

    # define optimizer
    group_params = init_group_params(net)
    lr = get_lr(lr_init=0, lr_end=0.0, lr_max=0.8, warmup_epochs=5, total_epochs=90, steps_per_epoch=step_size,
                lr_decay_mode="linear")
    lr = ms.Tensor(lr)
    opt = Momentum(group_params, lr, 0.9, loss_scale=1024)

    # define metrics
    metrics = {"acc"}

    # define model
    model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics, amp_level="O2",
                     boost_level="O0", keep_batchnorm_fp32=False)

    # define callback_1
    cb = [ms.TimeMonitor(data_size=step_size), ms.LossMonitor()]
    if get_rank_id() == 0:
        config_ck = ms.CheckpointConfig(save_checkpoint_steps=step_size * 10, keep_checkpoint_max=10)
        ck_cb = ms.ModelCheckpoint(prefix="resnet", directory="./checkpoint_stage_1", config=config_ck)
        cb += [ck_cb]

    # define callback_2: save weights for stage 2
    if get_rank_id() == 0:
        config_ck = ms.CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=40,
                                        saved_network=net)
        ck_cb = ms.ModelCheckpoint(prefix="resnet", directory="./checkpoint_stage_1/checkpoint_pca", config=config_ck)
        cb += [ck_cb]

    print("============== Starting Training ==============")
    model.train(70, ds_train, callbacks=cb, sink_size=step_size, dataset_sink_mode=True)

    if get_rank_id() == 0:
        print("============== Starting Testing ==============")
        eval_data_path = os.path.join(args.data_path, "val")
        ds_eval = create_dataset(dataset_path=eval_data_path, do_train=False, batch_size=256, target="Ascend")
        if ds_eval.get_dataset_size() == 0:
            raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

        acc = model.eval(ds_eval)
        print("============== {} ==============".format(acc))
