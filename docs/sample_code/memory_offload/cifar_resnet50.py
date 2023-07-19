# Copyright 2023 Huawei Technologies Co., Ltd
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
"""cifar_resnet50
This sample code is applicable to Ascend.
"""
import os
import random
import argparse
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication import init
from mindspore.nn import Momentum
from mindspore import train
from download import download
from resnet import resnet50


random.seed(1)
parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool,
                    default=False, help='Run distribute.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--device_target', type=str,
                    default="Ascend", help='Device choice Ascend or GPU')
parser.add_argument('--do_train', type=bool,
                    default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool,
                    default=False, help='Do eval or not.')
parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=10, help='Num classes.')
parser.add_argument('--checkpoint_path', type=str,
                    default=None, help='CheckPoint file path.')
parser.add_argument('--dataset_path', type=str,
                    default="./cifar-10-batches-bin", help='Dataset path.')

parser.add_argument('--memory_offload', type=str,
                    default="OFF", help='Memory offload.')
parser.add_argument('--offload_path', type=str,
                    default="./offload/", help='Offload path.')
parser.add_argument('--auto_offload', type=bool,
                    default=True, help='auto offload.')
parser.add_argument('--offload_param', type=str,
                    default="cpu", help='Offload param.')
parser.add_argument('--offload_cpu_size', type=str,
                    default="512GB", help='offload cpu size.')
parser.add_argument('--offload_disk_size', type=str,
                    default="1024GB", help='offload disk size.')
parser.add_argument('--host_mem_block_size', type=str,
                    default="1GB", help='host memory block size.')
parser.add_argument('--enable_pinned_mem', type=bool,
                    default=False, help='enable pinned mem.')
parser.add_argument('--enable_aio', type=bool,
                    default=False, help='enable aio.')
args_opt = parser.parse_args()

data_home = args_opt.dataset_path


url = "http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"

download(url, "./", kind="tar.gz", replace=True)


ms.set_context(mode=ms.GRAPH_MODE, device_target=args_opt.device_target)

if args_opt.memory_offload == "ON":
    ms.set_context(memory_offload="ON", max_device_memory="30GB")
    offload_config = {"offload_path": args_opt.offload_path, "auto_offload": args_opt.auto_offload,
                      "offload_param": args_opt.offload_param, "offload_cpu_size": args_opt.offload_cpu_size,
                      "offload_disk_size": args_opt.offload_disk_size,
                      "host_mem_block_size": args_opt.host_mem_block_size,
                      "enable_aio": args_opt.enable_aio, "enable_pinned_mem": args_opt.enable_pinned_mem}
    print("=====offload_config====\n", offload_config, flush=True)
    ms.set_offload_context(offload_config=offload_config)


if args_opt.device_target == "Ascend":
    device_id = int(os.getenv('DEVICE_ID', '0'))
    ms.set_context(device_id=device_id)


def create_dataset(repeat_num=1, training=True):
    """
    create data for next use such as training or inferring
    """
    assert os.path.exists(data_home), "the dataset path is invalid!"
    cifar_ds = ds.Cifar10Dataset(data_home)

    if args_opt.run_distribute:
        rank_id = int(os.getenv('RANK_ID'))
        rank_size = int(os.getenv('RANK_SIZE'))
        cifar_ds = ds.Cifar10Dataset(
            data_home, num_shards=rank_size, shard_id=rank_id)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = vision.RandomCrop(
        (32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    # interpolation default BILINEAR
    resize_op = vision.Resize((resize_height, resize_width))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = transforms.TypeCast(ms.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label")
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    cifar_ds = cifar_ds.shuffle(buffer_size=10)

    # apply batch operations
    cifar_ds = cifar_ds.batch(
        batch_size=args_opt.batch_size, drop_remainder=True)

    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_num)

    return cifar_ds


if __name__ == '__main__':
    # in this way by judging the mark of args, users will decide which function to use
    if not args_opt.do_eval and args_opt.run_distribute:
        ms.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                     all_reduce_fusion_config=[140])
        init()

    epoch_size = args_opt.epoch_size
    net = resnet50(args_opt.batch_size, args_opt.num_classes)
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)

    model = ms.Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})

    # as for train, users could use model.train
    if args_opt.do_train:
        dataset = create_dataset()
        batch_num = dataset.get_dataset_size()
        config_ck = train.CheckpointConfig(
            save_checkpoint_steps=batch_num, keep_checkpoint_max=35)
        ckpoint_cb = train.ModelCheckpoint(
            prefix="train_resnet_cifar10", directory="./", config=config_ck)
        loss_cb = train.LossMonitor()
        model.train(epoch_size, dataset, callbacks=[ckpoint_cb, loss_cb])

    # as for evaluation, users could use model.eval
    if args_opt.do_eval:
        if args_opt.checkpoint_path:
            param_dict = ms.load_checkpoint(args_opt.checkpoint_path)
            ms.load_param_into_net(net, param_dict)
        eval_dataset = create_dataset(training=False)
        res = model.eval(eval_dataset)
        print("result: ", res)
