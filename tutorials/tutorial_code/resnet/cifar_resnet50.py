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
'''cifar_resnet50
The sample can be run on Ascend 910 AI processor.
'''
import os
import random
import argparse
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops.functional as F
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from resnet import resnet50

random.seed(1)
parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=10, help='Num classes.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path.')
args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))

data_home = args_opt.dataset_path

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_id=device_id)

def create_dataset(repeat_num=1, training=True):
    """
    create data for next use such as training or infering
    """
    cifar_ds = ds.Cifar10Dataset(data_home)

    if args_opt.run_distribute:
        rank_id = int(os.getenv('RANK_ID'))
        rank_size = int(os.getenv('RANK_SIZE'))
        cifar_ds = ds.Cifar10Dataset(data_home, num_shards=rank_size, shard_id=rank_id)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4)) # padding_mode default CONSTANT
    random_horizontal_op = C.RandomHorizontalFlip()
    resize_op = C.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW()
    type_cast_op = C2.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    cifar_ds = cifar_ds.map(input_columns="label", operations=type_cast_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=c_trans)

    # apply shuffle operations
    cifar_ds = cifar_ds.shuffle(buffer_size=10)

    # apply batch operations
    cifar_ds = cifar_ds.batch(batch_size=args_opt.batch_size, drop_remainder=True)

    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_num)

    return cifar_ds

if __name__ == '__main__':
    # in this way by judging the mark of args, users will decide which function to use
    if not args_opt.do_eval and args_opt.run_distribute:
        context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
        auto_parallel_context().set_all_reduce_fusion_split_indices([140])
        init()

    epoch_size = args_opt.epoch_size
    net = resnet50(args_opt.batch_size, args_opt.num_classes)
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, is_grad=False, reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)

    model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})

    # as for train, users could use model.train
    if args_opt.do_train:
        dataset = create_dataset(epoch_size)
        batch_num = dataset.get_dataset_size()
        config_ck = CheckpointConfig(save_checkpoint_steps=batch_num, keep_checkpoint_max=35)
        ckpoint_cb = ModelCheckpoint(prefix="train_resnet_cifar10", directory="./", config=config_ck)
        loss_cb = LossMonitor()
        model.train(epoch_size, dataset, callbacks=[ckpoint_cb, loss_cb])

    # as for evaluation, users could use model.eval
    if args_opt.do_eval:
        if args_opt.checkpoint_path:
            param_dict = load_checkpoint(args_opt.checkpoint_path)
            load_param_into_net(net, param_dict)
        eval_dataset = create_dataset(1, training=False)
        res = model.eval(eval_dataset)
        print("result: ", res)
