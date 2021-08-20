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
resnet50_distributed_training_grad_accu
"""
import os
import mindspore.nn as nn
from mindspore import dtype as mstype
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.transforms.c_transforms as C
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import Tensor, context
from mindspore.nn import Momentum
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor
from resnet import resnet50
from model_accu import Model_ACCU
from cell_wrapper import TrainAccuStepsCell, VirtualDatasetCell

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_id=device_id) # set device_id
init()

def create_dataset(data_path, repeat_num=1, batch_size=32):     # pylint: disable=missing-docstring
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # get rank_id and rank_size
    rank_id = get_rank()
    rank_size = get_group_size()
    data_set = ds.Cifar10Dataset(data_path, num_shards=rank_size, shard_id=rank_id)

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set


class SoftmaxCrossEntropyExpand(nn.Cell):       # pylint: disable=missing-docstring
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = ops.RealDiv()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()
        self.eps = Tensor(1e-24, mstype.float32)

    def construct(self, logit, label):      # pylint: disable=missing-docstring
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)

        softmax_result_log = self.log(softmax_result + self.eps)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(ops.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss


def test_train_cifar(epoch_size=10):        # pylint: disable=missing-docstring
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,\
                                    gradients_mean=True, grad_accumulation_step=6)
    loss_cb = LossMonitor()
    data_path = os.getenv('DATA_PATH')
    batch_size = 32
    dataset = create_dataset(data_path, batch_size=batch_size)
    num_classes = 10
    net = resnet50(batch_size, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    net_with_loss = nn.WithLossCell(net, loss)
    net_with_loss = VirtualDatasetCell(net_with_loss)
    wrap_net = TrainAccuStepsCell(net_with_loss, opt)
    model = Model_ACCU(wrap_net)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=True)
