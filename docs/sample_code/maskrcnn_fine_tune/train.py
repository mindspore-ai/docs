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
# ===========================================================================
"""
Fine-tune process of MaskR-CNN.
"""

import os
import argparse
import ast

from mindspore import context, set_seed, load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.nn import Momentum

from src.maskrcnn.mask_rcnn_r50 import Mask_Rcnn_Resnet50
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.config import config
from src.dataset import create_new_dataset


set_seed(1)

parser = argparse.ArgumentParser(description="MaskRcnn training")
parser.add_argument("--do_train", type=ast.literal_eval,
                    default=True, help="Do train or not, default is true.")
parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
parser.add_argument("--pre_trained", type=str, default="resnet50_backbone.ckpt",
                    help="Pretrain file path.")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default is 0.")
args_opt = parser.parse_args()


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)
if __name__ == '__main__':
    print("Start train for maskrcnn!")
    rank = 0
    device_num = 1

    print("Start create dataset!")

    dataset = create_new_dataset(image_dir=config.coco_root, batch_size=config.batch_size,
                                 is_training=True, num_parallel_workers=8)
    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    net = Mask_Rcnn_Resnet50(config=config)
    net = net.set_train()

    load_path = args_opt.pre_trained
    if load_path != "":
        param_dict = load_checkpoint(load_path)
        if config.pretrain_epoch_size == 0:
            for item in list(param_dict.keys()):
                if not (item.startswith('backbone') or item.startswith('rcnn_mask')):
                    param_dict.pop(item)
        load_param_into_net(net, param_dict)

    loss = LossNet()
    lr = 0.001
    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                   weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    net_with_loss = WithLossCell(net, loss)
    net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path,
                                            'ckpt_' + str(rank) + '/')
        ckpoint_cb = ModelCheckpoint(prefix='mask_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=False)
