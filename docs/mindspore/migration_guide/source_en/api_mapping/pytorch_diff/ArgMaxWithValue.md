# Function Differences with torch.max

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/ArgMaxWithValue.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## torch.max

```python
torch.max(
    input,
    dim,
    keepdim=False,
    out=None
)
```

For more information, see [torch.max](https://pytorch.org/docs/1.5.0/torch.html#torch.max).

## mindspore.ops.ArgMaxWithValue

```python
class mindspore.ops.ArgMaxWithValue(
    axis=0,
    keep_dims=False
)(input_x)
```

For more information, see [mindspore.ops.ArgMaxWithValue](https://mindspore.cn/docs/api/en/r1.6/api_python/ops/mindspore.ops.ArgMaxWithValue.html#mindspore.ops.ArgMaxWithValue).

## Differences

PyTorch: Output tuple(max, index of max).

MindSpore: Output tuple(index of max, max).

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# Output tuple(index of max, max).
input_x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
argmax = ops.ArgMaxWithValue()
index, output = argmax(input_x)
print(index)
print(output)
# Out：
# 3
# 0.7

# Output tuple(max, index of max).
input_x = torch.tensor([0.0, 0.4, 0.6, 0.7, 0.1])
output, index = torch.max(input_x, 0)
print(index)
print(output)
# Out：
# tensor(3)
# tensor(0.7000)
```
