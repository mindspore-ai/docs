# 比较与torch.nn.functional.interpolate的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/interpolate.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.interpolate

```python
torch.nn.functional.interpolate(
    input,
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=None,
    recompute_scale_factor=None) -> Tensor
```

更多内容详见[torch.nn.functional.interpolate](https://pytorch.org/docs/1.8.1/nn.functional.html?highlight=interpolate#torch.nn.functional.interpolate)。

## mindspore.ops.interpolate

```python
mindspore.ops.interpolate(
    x,
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=None,
    recompute_scale_factor=None) -> Tensor
```

更多内容详见[mindspore.ops.interpolate](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.interpolate.html)。

## 使用方式

PyTorch：根据 `size` 或者 `scale_factor` 对数据进行上采样或者下采样。 `recompute_scale_factor` 控制是否重新计算用于插值计算的 `scale_factor` 。如果 `recompute_scale_factor` 为True，则必须传入 `scale_factor` ，并使用 `scale_factor` 计算输出大小。所计算的输出大小将用于推断插值的新比例。当 `scale_factor` 是浮点数时，由于舍入和精度问题，它可能与重新计算的比例不同。如果 `recompute_scale_factor` 为False，则直接使用 `size` 或 `scale_factor` 进行插值。插值方式可以选择'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'等六种模式。 `align_corners` 控制对齐坐标的对齐方式，对'linear' | 'bilinear' | 'bicubic' | 'trilinear'模式生效，默认为False。

MindSpore：和PyTorch实现功能基本一致，但是对于一些参数支持不完善，例如一些模式不能直接传入 `scale_factor` ，但可以通过设置 `recompute_scale_factor` 参数为True进行规避（当 `scale_factor` 为浮点数时，可能产生精度误差），具体差异如下。

| 分类 | 子类  | PyTorch | MindSpore | 差异 |
| ---- | ----- | ------- | --------- | ---- |
| 参数 | 参数1 | size     | size         | - |
|  | 参数2 | scale_factor | scale_factor  | 功能一致，目前仅支持'nearest'(5D)，'trilinear'和'area'模式直接传入缩放系数，对于不支持的模式可以通过设置 `recompute_scale_factor` 参数为True进行规避（当 `scale_factor` 为浮点数时，可能产生精度误差） |
|  | 参数3 | mode      | mode   | 功能一致，除上述六种模式外，MindSpore还支持'nearest-exact'模式 |
|  | 参数4 | align_corners | align_corners | 功能一致, 但在'bicubic'模式 `align_corners=False` 时，计算方式和TensorFlow相同，结果和PyTorch有差异 |
|  | 参数5 | recompute_scale_factor |   recompute_scale_factor    | - |
| 输入 | 单输入 | input      |  x  | 功能一致，参数名不同  |

## 差异分析与示例

### 代码示例1

> 使用默认'nearest'模式插值，传入 `size` ，两API实现同样的功能。

```python
# Pytorch
import torch
import numpy as np

x = torch.tensor(np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32))
output = torch.nn.functional.interpolate(input=x, size=6)
print(output.numpy())
# [[[1. 1. 2. 2. 3. 3.]
#   [4. 4. 5. 5. 6. 6.]]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32))
output = ops.interpolate(x, size=6, mode="nearest")
print(output)
# [[[1. 1. 2. 2. 3. 3.]
#   [4. 4. 5. 5. 6. 6.]]]
```

### 代码示例2

> 使用'bilinear'模式，传入 `scale_factor` 进行缩放，此模式下MindSpore不支持直接传入，但可以设置 `recompute_scale_factor` 参数为True进行规避（ `scale_factor` 为浮点数时，可能存在误差）。

```python
# Pytorch
import torch
import numpy as np

x = torch.tensor(np.array([[[[1, 2, 3], [4, 5, 6]]]]).astype(np.float32))
output = torch.nn.functional.interpolate(input=x, scale_factor=2, mode="bilinear", align_corners=True)
print(output.numpy())
# [[[[1.        1.4000001 1.8       2.2       2.6       3.       ]
#    [2.        2.4       2.8       3.1999998 3.6000001 4.       ]
#    [3.        3.4000003 3.8       4.2000003 4.6       5.       ]
#    [4.        4.4       4.8       5.2       5.6       6.       ]]]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.array([[[[1, 2, 3], [4, 5, 6]]]]).astype(np.float32))
output = ops.interpolate(x, scale_factor=2, recompute_scale_factor=True, mode="bilinear", align_corners=True)
print(output)
# [[[[1.        1.4       1.8       2.2       2.6       3.       ]
#    [2.        2.4       2.8000002 3.2       3.6       4.       ]
#    [3.        3.4       3.8000002 4.2       4.6       5.       ]
#    [4.        4.4       4.8       5.2       5.6       6.       ]]]]
```
