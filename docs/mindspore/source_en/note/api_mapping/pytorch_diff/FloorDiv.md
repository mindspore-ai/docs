# Function Differences with torch.floor_divide

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/FloorDiv.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [mindspore.ops.FloorDiv](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.FloorDiv.html#mindspore.ops.FloorDiv).

## Differences

PyTorch: The result is rounded toward 0, not rounded down to the nearest integer. For example, if the division is -0.9, the result will be 0 after rounding.

MindSpore: The result is rounded down by floor. For example, if the division is -0.9, the result will be -1 after rounding.

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, the output will be rounded toward the floor, so, after division, the output -0.33 will be rounded to -1.
input_x = ms.Tensor(np.array([2, 4, -1]), ms.int32)
input_y = ms.Tensor(np.array([3, 3, 3]), ms.int32)
floor_div = ops.FloorDiv()
output = floor_div(input_x, input_y)
print(output)
# Out：
# [ 0  1 -1]

# In torch, the output will be rounded toward 0, so, after division, the output -0.33 will be rounded to 0.
input_x = torch.tensor(np.array([2, 4, -1]))
input_y = torch.tensor(np.array([3, 3, 3]))
output = torch.floor_divide(input_x, input_y)
print(output)
# Out：
# tensor([0, 1, 0])
```