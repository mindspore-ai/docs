# Function Differences with torch.nn.functional.interpolate

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/interpolate.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [torch.nn.functional.interpolate](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.interpolate).

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

For more information, see [mindspore.ops.interpolate](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.interpolate.html).

## Usage

PyTorch: Data is upsampled or downsampled based on `size` or `scale_factor`. The `recompute_scale_factor` controls whether the `scale_factor` used for interpolation calculation is re-calculated. If `recompute_scale_factor` is True, `scale_factor` must be passed in and the output size is calculated using `scale_factor`. The calculated output size will be used to infer the new ratio for interpolation. When `scale_factor` is a floating point number, it may be different from the re-calculated ratio due to rounding and precision issues. If `recompute_scale_factor` is False, interpolation is performed directly using `size` or `scale_factor`. Interpolation can be performed using one of six modes: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'. The `align_corners` controls the alignment of the coordinates and is effective for 'linear' | 'bilinear' | 'bicubic' | 'trilinear' modes, with a default value of False.

MindSpore: The functionality is basically the same as PyTorch, but support for some parameters is not complete, such as some modes cannot directly pass in `scale_factor`, but can be circumvented by setting the `recompute_scale_factor` parameter to True (when `scale_factor` is a floating point number, accuracy errors may occur), and the specific differences are as follows.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ---- |
| Parameter | Parameter 1 | size | size | - |
|  | Parameter 2 | scale_factor | scale_factor | Function is consistent. Currently only supports 'nearest' (5D), 'trilinear' and 'area' modes directly pass in `scale_factor`. For unsupported modes, you can bypass by setting `recompute_scale_factor` parameter to True (when `scale_factor` is a floating-point number, there may be precision errors) |
|  | Parameter 3 | mode | mode | Function is consistent, in addition to the above six modes, MindSpore also supports 'nearest-exact' mode |
|  | Parameter 4 | align_corners | align_corners | Function is consistent, but in 'bicubic' mode `align_corners=False`, the calculation method is the same as TensorFlow, and the results are different from PyTorch |
|  | Parameter 5 | recompute_scale_factor | recompute_scale_factor | - |
| Input | Single input | input | x | Same function, different parameter names |

## Difference Analysis and Examples

### Code Example 1

> Using the default 'nearest' mode interpolation, pass `size` in and the two APIs achieve the same function.

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

### Code Example 2

> Using the 'bilinear' mode, scale with `scale_factor`. This mode is not directly supported by MindSpore, but the error can be avoided by setting the `recompute_scale_factor` parameter to True (when `scale_factor` is a floating point, there may be some inaccuracies).

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
