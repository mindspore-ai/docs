# Differences with torchvision.ops.deform_conv2d

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/deform_conv2d.md)

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

For more information, see [torchvision.ops.deform_conv2d](https://pytorch.org/vision/0.9/transforms.html#torchvision.ops.deform_conv2d.html).

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

For more information, see [mindspore.ops.deformable_conv2d](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.deformable_conv2d.html).

## Differences

PyTorch: Parameters offsets is a 4D tensor of x-y coordinates offset. With the format "NCHW", the shape is $\left(batch, deformable\underline{ }groups × H_{\text {f }} × W_{\text {f }} × 2, H_{\text {out }}, W_{\text {out }}\right)$. Note the C dimension is stored in the order of $\left(deformable\underline{ }groups, H_{\text {f }}, W_{\text {f }}, \left(offset\underline{ }y, offset\underline{ }x\right)\right)$. Parameters mask is a 4D tensor of mask. With the format "NCHW", the shape is $\left(batch, deformable\underline{ }groups × H_{\text {f }} × W_{\text {f }} × 1, H_{\text {out }}, W_{\text {out }}\right)$. Note the C dimension is stored in the order of $\left(deformable\underline{ }groups, H_{f}, W_{f}, mask\right)$.

MindSpore: Parameters offsets is a 4D tensor of x-y coordinates offset and mask. With the format "NCHW", the shape is $\left(batch, 3 × deformable\underline{ }groups × H_{\text {f }} × W_{\text {f }}, H_{\text {out }}, W_{\text {out }}\right)$. Note the C dimension is stored in the order of $\left(\left(offset\underline{ }x, offset\underline{ }y, mask\right), deformable\underline{ }groups, H_{f}, W_{f}\right)$.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | input     | x         | Same function, different parameter names |
|      | Parameter 2 | offset   | offsets | MindSpore parameters offsets is a 4D tensor of x-y coordinates offset and mask |
|      | Parameter 3 | weight     | weight         | - |
|      | Parameter 4 | -     | kernel_size         | Pytorch does not have this parameter |
|      | Parameter 5 | mask     | -         | MindSpore does not have this parameter |
|      | Parameter 6 | bias     | bias         | - |
|      | Parameter 7 | stride     | strides         | Same function, different parameter names |
|      | Parameter 8 | padding     | padding         | - |
|      | Parameter 9 | dilations     | dilations         | - |
|      | Parameter 10 | -     | groups         | Pytorch does not have this parameter |
|      | Parameter 11 | -     | deformable_groups         | Pytorch does not have this parameter |
|      | Parameter 12 | -     | modulated         | Pytorch does not have this parameter |

### Code Example

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
