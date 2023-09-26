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

"""Memory Offload Example"""

import os
import argparse
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops
from mindspore.communication import init, get_rank, get_group_size

parser = argparse.ArgumentParser(description='Memory offload')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
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

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
ms.set_context(max_device_memory="1GB")
if args_opt.memory_offload == "ON":
    ms.set_context(memory_offload="ON")
    offload_config = {"offload_path": args_opt.offload_path, "auto_offload": args_opt.auto_offload,
                      "offload_param": args_opt.offload_param, "offload_cpu_size": args_opt.offload_cpu_size,
                      "offload_disk_size": args_opt.offload_disk_size,
                      "host_mem_block_size": args_opt.host_mem_block_size,
                      "enable_aio": args_opt.enable_aio, "enable_pinned_mem": args_opt.enable_pinned_mem}
    print("=====offload_config====\n", offload_config, flush=True)
    ms.set_offload_context(offload_config=offload_config)
init()
ms.set_seed(1)

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Dense(16, 10)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.avgpool(x).squeeze()
        logits = self.dense(x)
        return logits

net = Network()

def create_dataset(batch_size):
    """create dataset"""
    dataset_path = os.getenv("DATA_PATH")
    rank_id = get_rank()
    rank_size = get_group_size()
    cifar_ds = ds.Cifar10Dataset(dataset_path, num_shards=rank_size, shard_id=rank_id)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = ds.vision.RandomCrop(
        (32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = ds.vision.RandomHorizontalFlip()
    # interpolation default BILINEAR
    resize_op = ds.vision.Resize((resize_height, resize_width))
    rescale_op = ds.vision.Rescale(rescale, shift)
    normalize_op = ds.vision.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = ds.vision.HWC2CHW()
    type_cast_op = ds.transforms.TypeCast(ms.int32)

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
        batch_size=batch_size, drop_remainder=True)
    return cifar_ds

data_set = create_dataset(args_opt.batch_size)
optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    """forward propagation"""
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

i = 0
step = 10
for image, label in data_set:
    (loss_value, _), grads = grad_fn(image, label)
    grads = grad_reducer(grads)
    optimizer(grads)
    print("step: %s, loss is %s" % (i, loss_value))
    if i >= step:
        break
    i += 1
