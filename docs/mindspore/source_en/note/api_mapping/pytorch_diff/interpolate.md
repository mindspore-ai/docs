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

For more information, see [torch.nn.functional.interpolate](https://pytorch.org/docs/1.8.1/nn.functional.html?highlight=interpolate#torch.nn.functional.interpolate).

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

For more information, see [mindspore.ops.interpolate](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.interpolate.html).

## Usage

PyTorch: The data is upsampled or downsampled according to `size` or `scale_factor`, and there are various interpolation options for sampling.

MindSpore: MindSpore API basically implements the same function as PyTorch, but currently only supports 'linear' and 'bilinear' modes for sampling data.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | input      | x         | Same function, different parameter names  |
|  | Parameter 2 | size      | sizes         | Same function, different parameter names  |
|  | Parameter 3 | scale_factor      | scales         | Same function, different parameter names  |
|  | Parameter 4 | mode      | mode         | -  |
|  | Parameter 5 | align_corners      | coordinate_transformation_mode         | MindSpore is also specified as "half_pixel" and "asymmetric"  |
|  | Parameter 6 | recompute_scale_factor      | -         | PyTorch can calculate scale_factor based on output size and input size, which is not currently supported by MindSpore.  |
|  | Parameter 7 | -      | roi         | Reserved input, effective in "crop_and_resize" coordinate transformation mode, currently not available |

## Difference Analysis and Examples

### Code Example 1

> Using linear interpolation. The two APIs achieve the same function.

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

### Code Example 2

> Using bilinear interpolation. The two APIs achieve the same function.

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

