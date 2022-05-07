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
"""
Resnet50_distributed_training
"""
import os
import mindspore.nn as nn
from mindspore import dtype as mstype
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.transforms.c_transforms as C
from mindspore.communication import init, get_rank
from mindspore import Tensor, Model, ParallelMode, set_context, GRAPH_MODE, set_auto_parallel_context
from mindspore.nn import Momentum
from mindspore.train.callback import LossMonitor
from resnet import resnet50

device_id = int(os.getenv('DEVICE_ID'))
set_context(mode=GRAPH_MODE, device_target="Ascend")
set_context(device_id=device_id) # set device_id
init()
ds.config.set_seed(1000) # set dataset seed to make sure than all cards read the same data

def create_dataset(data_path, repeat_num=1, batch_size=32, slice_h_num=1, slice_w_num=1):     # pylint: disable=missing-docstring
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # get rank_id and rank_size
    rank_id = get_rank()

    # create a full dataset before slice
    data_set = ds.Cifar10Dataset(data_path, shuffle=True)

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    # in random map func, using num_parallel_workers=1 to avoid the dataset random seed not work.
    data_set = data_set.map(operations=c_trans, input_columns="image", num_parallel_workers=1)
    # slice image
    slice_patchs_img_op = vision.SlicePatches(slice_h_num, slice_w_num)
    img_cols = ['img' + str(x) for x in range(slice_h_num * slice_w_num)]
    data_set = data_set.map(operations=slice_patchs_img_op, input_columns="image", output_columns=img_cols,
                            column_order=[img_cols[rank_id % (slice_h_num * slice_w_num)], "label"])
    # change hwc to chw
    data_set = data_set.map(operations=changeswap_op, input_columns=img_cols[rank_id % (slice_h_num * slice_w_num)])
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
    set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    slice_h_num = 1
    slice_w_num = 8
    batch_size = 256
    set_auto_parallel_context(dataset_strategy=(((1, 1, slice_h_num, slice_w_num), (1,))))
    loss_cb = LossMonitor()
    data_path = os.getenv('DATA_PATH')
    dataset = create_dataset(data_path, batch_size=batch_size, slice_h_num=slice_h_num, slice_w_num=slice_w_num)
    num_classes = 10
    net = resnet50(batch_size, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=True)
