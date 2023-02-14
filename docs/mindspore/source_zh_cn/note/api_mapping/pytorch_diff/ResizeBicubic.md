# 比较与torch.nn.Upsample的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ResizeBicubic.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Upsample

```text
torch.nn.Upsample(
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=None
)(input) -> Tensor
```

更多内容详见[torch.nn.Upsample](https://pytorch.org/docs/1.8.1/generated/torch.nn.Upsample.html#torch.nn.Upsample)。

## mindspore.ops.ResizeBicubic

```text
class mindspore.ops.ResizeBicubic(
    align_corners=False,
    half_pixel_centers=False
)(images, size) -> Tensor
```

更多内容详见[mindspore.ops.ResizeBicubic](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ResizeBicubic.html#mindspore.ops.ResizeBicubic)。

## 差异对比

PyTorch：当mode为 `bicubic` 时，对输入图像进行双三次插值(Bicubic interpolation)。

MindSpore：MindSpore此API实现功能与PyTorch `bicubic` mode存在一定差别。MindSpore align_corners=True，half_pixel_centers=False时与PyTorch align_corners=True实现基本一致；MindSpore align_corners=False，half_pixel_centers=True时与PyTorch align_corner=False实现存在两处差别：

- 在该模式下，MindSpore在bicubic interpolation中使用的系数 `a=-0.5`，而PyTorch中 `a=-0.75` ；
- MindSpore Tensor之外采样位置的权重被置为0并重新归一化，故所有权重之和为1.0，PyTorch则将Tensor之外的采样改为对边界位置进行采样，其权重保持不变，故边界位置将被赋予更大的权重。

因此，在第二种模式下，MindSpore与PyTorch的计算结果存在差异。

| 分类 | 子类  | PyTorch   | MindSpore | 差异                                                         |
| ---- | ----- | --------- | --------- | ------------------------------------------------------------ |
| 输入  | 输入1  | input      | images    |   -                                                          |
|      | 输入2 |    -       | size      |  -                                                           |
| 参数  | 参数1 | size       |  -       |                                -                               |
|      | 参数2 | scale_factor |  -     |      PyTorch计算输出size的乘数，MindSpore无此功能                   |
|      | 参数3 | mode    |       -     | PyTorch决定插值算法的参数，mode='bicubic'时使用双三次插值算法          |
|      | 参数4 | align_corners | align_corners   |       -                                               |
|      | 参数5 |   -     | half_pixel_centers | MindSpore是否使用半像素中心对齐的标志，PyTorch align_corners=False时使用半像素中心对齐 |

### 代码示例1

```python
# PyTorch
import numpy as np
import torch

x_np = np.array([1, 2, 3, 4]).astype(np.float32).reshape(1, 1, 2, 2)
size = (1, 4)
upsample = torch.nn.Upsample(size=size, mode='bicubic', align_corners=True)
out = upsample(torch.from_numpy(x_np))
print(out.detach().numpy())
# [[[[1., 1.3148143, 1.6851856, 2.]]]]

# MindSpore
import mindspore
from mindspore import Tensor
from mindspore.ops as ops

resize_bicubic_op = ops.ResizeBicubic(align_corners=True,half_pixel_centers=False)
images = Tensor(x_np)
size = Tensor(size, mindspore.int32)
output = resize_bicubic_op(images, size)
print(output.asnumpy())
# [[[[1., 1.3144622, 1.6855378, 2.]]]]
```

### 代码示例2

```python
# PyTorch
import numpy as np
import torch

x_np = np.array([1, 2, 3, 4]).astype(np.float32).reshape(1, 1, 2, 2)
size = (1, 4)
upsample = torch.nn.Upsample(size=size, mode='bicubic', align_corners=False)
out = upsample(torch.from_numpy(x_np))
print(out.detach().numpy())
# [[[[0.70703125, 1.0390625, 1.5859375, 1.9179688]]]]

# MindSpore
import mindspore
from mindspore import Tensor
from mindspore.ops as ops

resize_bicubic_op = ops.ResizeBicubic(align_corners=False, half_pixel_centers=True)
images = Tensor(x_np)
size = Tensor(size, mindspore.int32)
output = resize_bicubic_op(images, size)
print(output.asnumpy())
# [[[[1.9117649, 2.2071428, 2.7928572, 3.0882356]]]]
```