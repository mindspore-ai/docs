# 比较与torchvision.ops.deform_conv2d的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/deform_conv2d.md)

## torchvision.ops.deform_conv2d

```text
class torchvision.ops.deform_conv2d(
    input,
    offset,
    weight,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilations=(1, 1),
    mask=None
)
```

更多内容详见[torchvision.ops.deform_conv2d](https://pytorch.org/vision/0.9/transforms.html#torchvision.ops.deform_conv2d.html).

## mindspore.ops.deformable_conv2d

```text
class mindspore.ops.deformable_conv2d(
    x,
    weight,
    offsets,
    kernel_size,
    strides,
    padding,
    bias=None,
    dilations=(1, 1, 1, 1),
    groups=1,
    deformable_groups=1,
    modulated=True
)
```

更多内容详见[mindspore.ops.deformable_conv2d](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.deformable_conv2d.html).

## 差异对比

PyTorch: 参数offsets是一个四维Tensor，存储x和y坐标的偏移。数据格式为“NCHW”，shape为$\left(batch, deformable\underline{ }groups × H_{\text {f }} × W_{\text {f }} × 2, H_{\text {out }}, W_{\text {out }}\right)$，注意其中C维度的存储顺序为$\left(deformable\underline{ }groups, H_{\text {f }}, W_{\text {f }}, \left(offset\underline{ }y, offset\underline{ }x\right)\right)$。参数mask是一个四维Tensor，存储可变形卷积的输入掩码mask。数据格式为“NCHW”，shape为$\left(batch, deformable\underline{ }groups × H_{\text {f }} × W_{\text {f }} × 1, H_{\text {out }}, W_{\text {out }}\right)$，注意其中C维度的存储顺序为$\left(deformable\underline{ }groups, H_{f}, W_{f}, mask\right)$。

MindSpore: 一个四维Tensor，存储x和y坐标的偏移，以及可变形卷积的输入掩码mask。数据格式为“NCHW”，shape为$\left(batch, 3 × deformable\underline{ }groups × H_{\text {f }} × W_{\text {f }}, H_{\text {out }}, W_{\text {out }}\right)$，注意其中C维度的存储顺序为$\left(\left(offset\underline{ }x, offset\underline{ }y, mask\right), deformable\underline{ }groups, H_{f}, W_{f}\right)$。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | input   | x        |  功能一致，参数名不同 |
|      | 参数2 | offset   | offsets        | MindSpore的offsets参数包含PyTorch的offset和mask两个参数 |
|      | 参数3 | weight   | weight        | - |
|      | 参数4 | -  | kernel_size        | Pytorch无此参数 |
|      | 参数5 | mask   | -        | MindSpore无此参数 |
|      | 参数6 | bias   | bias        | - |
|      | 参数7 | stride   | strides        | 功能一致，参数名不同 |
|      | 参数8 | padding   | padding        | - |
|      | 参数9 | dilations   | dilations        | - |
|      | 参数10 | -  | groups        | Pytorch无此参数 |
|      | 参数11 | -  | deformable_groups        | Pytorch无此参数 |
|      | 参数12 | -  | modulated        | Pytorch无此参数 |

### 代码示例

```python
# PyTorch
import torch
from torch import tensor
import numpy as np
from torchvision.ops import deform_conv2d
np.random.seed(1)
kh, kw = 1, 1
batch = 1
deformable_groups = 1
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1
pad_h, pad_w = 0, 0
x_h, x_w = 1, 2
out_h = (x_h + 2 * pad_h - dilation_h * (kh - 1) - 1) // stride_h + 1
out_w = (x_w + 2 * pad_w - dilation_w * (kw - 1) - 1) // stride_w + 1

x = np.random.randn(batch, 64, x_h, x_w).astype(np.float32)
weight = np.random.randn(batch, 64, kh, kw).astype(np.float32)
offsets_x = np.random.randn(batch, 1, deformable_groups, kh, kw, out_h, out_w).astype(np.float32)
offsets_y = np.random.randn(batch, 1, deformable_groups, kh, kw, out_h, out_w).astype(np.float32)
mask = np.random.randn(batch, 1, deformable_groups, kh, kw, out_h, out_w).astype(np.float32)

offsets = np.concatenate((offsets_y, offsets_x), axis=1)
offsets = offsets.transpose(0, 2, 3, 4, 1, 5, 6)
offsets = offsets.reshape((batch, 2 * deformable_groups * kh * kw, out_h, out_w))
mask = mask.transpose(0, 2, 3, 4, 1, 5, 6)
mask = mask.reshape((batch, 1 * deformable_groups * kh * kw, out_h, out_w))
x = torch.from_numpy(x.copy().astype(np.float32))
weight = torch.from_numpy(weight.copy().astype(np.float32))
offsets = torch.from_numpy(offsets.copy().astype(np.float32))
mask = torch.from_numpy(mask.copy().astype(np.float32))
output = deform_conv2d(x, offsets, weight, stride=(stride_h, stride_w), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), mask=mask)
print(output)
# tensor([[[[-0.0022,  0.0000]]]])

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np
from mindspore.ops import deformable_conv2d
import mindspore.ops as ops
np.random.seed(1)
kh, kw = 1, 1
batch = 1
deformable_groups = 1
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1
pad_h, pad_w = 0, 0
x_h, x_w = 1, 2
out_h = (x_h + 2 * pad_h - dilation_h * (kh - 1) - 1) // stride_h + 1
out_w = (x_w + 2 * pad_w - dilation_w * (kw - 1) - 1) // stride_w + 1

x = np.random.randn(batch, 64, x_h, x_w).astype(np.float32)
weight = np.random.randn(batch, 64, kh, kw).astype(np.float32)
offsets_x = np.random.randn(batch, 1, deformable_groups, kh, kw, out_h, out_w).astype(np.float32)
offsets_y = np.random.randn(batch, 1, deformable_groups, kh, kw, out_h, out_w).astype(np.float32)
mask = np.random.randn(batch, 1, deformable_groups, kh, kw, out_h, out_w).astype(np.float32)

offsets = np.concatenate((offsets_x, offsets_y, mask), axis=1)
offsets = offsets.reshape((batch, 3 * deformable_groups * kh * kw, out_h, out_w))

x = Tensor(x)
weight = Tensor(weight)
offsets = Tensor(offsets)
output = ops.deformable_conv2d(x, weight, offsets, (kh, kw), (1, 1, stride_h, stride_w,), (pad_h, pad_h, pad_w, pad_w), dilations=(1, 1, dilation_h, dilation_w))
print(output)
# [[[[-0.00220442  0.        ]]]]
```
