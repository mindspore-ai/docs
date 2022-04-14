# Function Differences with torch.clamp

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/clip_by_value.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.clamp

```python
torch.clamp(
    input,
    min,
    max,
    out=None
)
```

For more information, see [torch.clamp](https://pytorch.org/docs/1.5.0/torch.html#torch.clamp).

## mindspore.ops.clip_by_value

```python
mindspore.ops.clip_by_value(
    x,
    clip_value_min,
    clip_value_max
)
```

For more information, see [mindspore.ops.clip_by_value](https://mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.clip_by_value.html#mindspore.ops.clip_by_value).

## Differences

PyTorch: Clamps all elements in input into the range [ min, max ] and return a resulting tensor. Supports specifying one of two parameters ‘min’, ‘max’.

MindSpore：Limits the value of 'x' to a range, whose lower limit is ‘clip_value_min’ and upper limit is ‘clip_value_max’. The two parameters ‘clip_value_min’, ‘clip_value_max’ are required.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

min_value = Tensor(5, mindspore.float32)
max_value = Tensor(20, mindspore.float32)
x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
output = ops.clip_by_value(x, min_value, max_value)
print(output)
# Out：
# [[ 5. 20.  5.  7.]
#  [ 5. 11.  6. 20.]]

a = torch.randn(4)
print(a)
# Out：
#tensor([-1.7120,  0.1734, -0.0478, -0.0922])
print(torch.clamp(a, min=-0.5, max=0.5))
# Out：
# tensor([-0.5000,  0.1734, -0.0478, -0.0922])

a = torch.randn(4)
print(a)
# Out：
# tensor([-0.0299, -2.3184,  2.1593, -0.8883])
print(torch.clamp(a, min=0.5))
# Out：
# tensor([ 0.5000,  0.5000,  2.1593,  0.5000])

a = torch.randn(4)
print(a)
# Out：
# tensor([ 0.7753, -0.4702, -0.4599,  1.1899])
print(torch.clamp(a, max=0.5))
# Out：
# tensor([ 0.5000, -0.4702, -0.4599,  0.5000])
```