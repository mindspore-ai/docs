# 比较与torch.nn.functional.interpolate的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/interpolate.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

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
    roi=None,
    scales=None,
    sizes=None,
    coordinate_transformation_mode="align_corners",
    mode="linear") -> Tensor
```

更多内容详见[mindspore.ops.interpolate](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.interpolate.html)。

## 使用方式

PyTorch：根据 `size` 或者 `scale_factor` 对数据进行上采样或者下采样，采样有多种插值方式可以选择。

MindSpore：和PyTorch实现功能基本一致，不过当前仅支持'linear'和'bilinear'两种模式对数据进行采样。

| 分类 | 子类  | PyTorch | MindSpore | 差异 |
| ---- | ----- | ------- | --------- | ---- |
| 参数 | 参数1 | input      | x         | 功能一致，参数名不同  |
|  | 参数2 | size      | sizes         | 功能一致，参数名不同  |
|  | 参数3 | scale_factor      | scales         | 功能一致，参数名不同  |
|  | 参数4 | mode      | mode         | -  |
|  | 参数5 | align_corners      | coordinate_transformation_mode         | MindSpore还指定为"half_pixel"和"asymmetric"  |
|  | 参数6 | recompute_scale_factor      | -         | PyTorch可根据输出size和输入size计算scale_factor，MindSpore目前不支持此功能  |
|  | 参数7 | -      | roi         | 保留输入，在"crop_and_resize"坐标变换模式下生效，当前还未开放 |

## 差异分析与示例

### 代码示例1

> 使用线性插值，两API实现同样的功能。

```python
# Pytorch
import torch
import numpy as np

x = torch.tensor(np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32))
output = torch.nn.functional.interpolate(input=x, size=(6,), scale_factor=None, mode="linear", align_corners=True)
print(output.numpy())
# [[[1.        1.4000001 1.8       2.2       2.6       3.       ]
#   [4.        4.4       4.8       5.2       5.6       6.       ]]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32))
output = ops.interpolate(x=x, roi=None, scales=None, sizes=(6,), coordinate_transformation_mode="align_corners", mode="linear")
print(output)
# [[[1.  1.4 1.8 2.2 2.6 3. ]
#   [4.  4.4 4.8 5.2 5.6 6. ]]]
```

### 代码示例2

> 使用双线性插值，两API实现同样的功能。

```python
# Pytorch
import torch
import numpy as np

x = torch.tensor(np.array([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]]).astype(np.float32))
output = torch.nn.functional.interpolate(input=x, size=(5, 5), scale_factor=None, mode="bilinear", align_corners=True)
print(output.numpy())
# [[[[1. 2. 3. 4. 5.]
#    [1. 2. 3. 4. 5.]
#    [1. 2. 3. 4. 5.]
#    [1. 2. 3. 4. 5.]
#    [1. 2. 3. 4. 5.]]]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.array([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]]).astype(np.float32))
output = ops.interpolate(x=x, roi=None, scales=None, sizes=(5, 5), coordinate_transformation_mode="align_corners", mode="bilinear")
print(output)
# [[[[1. 2. 3. 4. 5.]
#    [1. 2. 3. 4. 5.]
#    [1. 2. 3. 4. 5.]
#    [1. 2. 3. 4. 5.]
#    [1. 2. 3. 4. 5.]]]]
```
