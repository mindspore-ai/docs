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
"""MaskRcnn feature pyramid network."""

import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore.common import Tensor, dtype as mstype
from mindspore.common.initializer import initializer


def bias_init_zeros(shape):
    """Bias init method."""
    return Tensor(np.array(np.zeros(shape).astype(np.float32)), dtype=mstype.float32)

def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = initializer("XavierUniform", shape=shape, dtype=mstype.float32)
    shape_bias = (out_channels,)
    biass = bias_init_zeros(shape_bias)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=biass)

class FeatPyramidNeck(nn.Cell):
    """
    Feature pyramid network cell, usually uses as network neck.

    Applies the convolution on multiple, input feature maps
    and output feature map with same channel size. if required num of
    output larger then num of inputs, add extra maxpooling for further
    downsampling;

    Args:
        in_channels (tuple) - Channel size of input feature maps.
        out_channels (int) - Channel size output.
        num_outs (int) - Num of output features.

    Returns:
        Tuple, with tensors of same channel size.

    Examples:
        neck = FeatPyramidNeck([100,200,300], 50, 4)
        input_data = (normal(0,0.1,(1,c,1280//(4*2**i), 768//(4*2**i)),
                      dtype=np.float32) \
                      for i, c in enumerate(config.fpn_in_channels))
        x = neck(input_data)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs):
        super(FeatPyramidNeck, self).__init__()
        self.num_outs = num_outs
        self.in_channels = in_channels
        self.fpn_layer = len(self.in_channels)

        assert not self.num_outs < len(in_channels)

        self.lateral_convs_list_ = []
        self.fpn_convs_ = []

        for _, channel in enumerate(in_channels):
            l_conv = _conv(channel, out_channels, kernel_size=1, stride=1,
                           padding=0, pad_mode='valid').to_float(mstype.float16)
            fpn_conv = _conv(out_channels, out_channels, kernel_size=3, stride=1,
                             padding=0, pad_mode='same').to_float(mstype.float16)
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        self.interpolate1 = ops.ResizeBilinear((48, 80))
        self.interpolate2 = ops.ResizeBilinear((96, 160))
        self.interpolate3 = ops.ResizeBilinear((192, 320))
        self.cast = ops.Cast()
        self.maxpool = ops.MaxPool(kernel_size=1, strides=2, pad_mode="same")

    def construct(self, inputs):
        """Feature pyramid network cell, usually uses as network neck."""
        x = ()
        for i in range(self.fpn_layer):
            x += (self.lateral_convs_list[i](inputs[i]),)

        y = (x[3],)
        y = y + (x[2] + self.cast(self.interpolate1(y[self.fpn_layer - 4]), mstype.float16),)
        y = y + (x[1] + self.cast(self.interpolate2(y[self.fpn_layer - 3]), mstype.float16),)
        y = y + (x[0] + self.cast(self.interpolate3(y[self.fpn_layer - 2]), mstype.float16),)

        z = ()
        for i in range(self.fpn_layer - 1, -1, -1):
            z = z + (y[i],)

        outs = ()
        for i in range(self.fpn_layer):
            outs = outs + (self.fpn_convs_list[i](z[i]),)

        for i in range(self.num_outs - self.fpn_layer):
            outs = outs + (self.maxpool(outs[3]),)
        return outs
