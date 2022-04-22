# Function Differences with torch.nn.functional.normalize

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/L2Normalize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.normalize

```python
torch.nn.functional.normalize(
    input,
    p=2,
    dim=1,
    eps=1e-12,
    out=None
)
```

For more information, see [torch.nn.functional.normalize](https://pytorch.org/docs/1.5.0/nn.functional.html#torch.nn.functional.normalize).

## mindspore.ops.L2Normalize

```python
class mindspore.ops.L2Normalize(
    axis=0,
    epsilon=1e-4
)(input_x)
```

For more information, see [mindspore.ops.L2Normalize](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.L2Normalize.html#mindspore.ops.L2Normalize).

## Differences

PyTorch: Supports using the LP paradigm by specifying the parameter `p`.

MindSpore：Only L2 paradigm is supported.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, you can directly pass data into the function, and the default dimension is 0.
l2_normalize = ops.L2Normalize()
input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
output = l2_normalize(input_x)
print(output)
# Out：
# [0.2673 0.5345 0.8018]

# In torch, parameter p should be set to determine it is a lp normalization, and the default dimension is 1.
input_x = torch.tensor(np.array([1.0, 2.0, 3.0]))
outputL2 = torch.nn.functional.normalize(input=input_x, p=2, dim=0)
outputL3 = torch.nn.functional.normalize(input=input_x, p=3, dim=0)
print(outputL2)
print(outputL3)
# Out：
# tensor([0.2673, 0.5345, 0.8018], dtype=torch.float64)
# tensor([0.3029, 0.6057, 0.9086], dtype=torch.float64)
```