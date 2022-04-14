# Function Differences with torch.floor_divide

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/FloorDiv.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## torch.floor_divide

```python
torch.floor_divide(
    input,
    other,
    out=None
)
```

For more information, see [torch.floor_divide](https://pytorch.org/docs/1.5.0/torch.html#torch.floor_divide).

## mindspore.ops.FloorDiv

```python
class mindspore.ops.FloorDiv(*args, **kwargs)(
    input_x,
    input_y
)
```

For more information, see [mindspore.ops.FloorDiv](https://mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.FloorDiv.html#mindspore.ops.FloorDiv).

## Differences

PyTorch: The output will be rounded toward 0 rather than the floor.

MindSpore: The output will be rounded exactly toward floor.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, the output will be rounded toward the floor, so, after division, the output -0.33 will be rounded to -1.
input_x = Tensor(np.array([2, 4, -1]), mindspore.int32)
input_y = Tensor(np.array([3, 3, 3]), mindspore.int32)
floor_div = ops.FloorDiv()
output = floor_div(input_x, input_y)
print(output)
# Out：
# [0 1 -1]

# In torch, the output will be rounded toward 0, so, after division, the output -0.33 will be rounded to 0.
input_x = torch.tensor(np.array([2, 4, -1]))
input_y = torch.tensor(np.array([3, 3, 3]))
output = torch.floor_divide(input_x, input_y)
print(output)
# Out：
# tensor([0, 1, 0])
```