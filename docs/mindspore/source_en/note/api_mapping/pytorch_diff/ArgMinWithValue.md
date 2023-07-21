# Function Differences with torch.min

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ArgMinWithValue.md)

## torch.min

```python
torch.min(
    input,
    dim,
    keepdim=False,
    out=None
)
```

For more information, see [torch.min](https://pytorch.org/docs/1.5.0/torch.html#torch.min).

## mindspore.ops.ArgMinWithValue

```python
class mindspore.ops.ArgMinWithValue(
    axis=0,
    keep_dims=False
)(input_x)
```

For more information, see [mindspore.ops.ArgMinWithValue](https://mindspore.cn/docs/en/r1.8/api_python/ops/mindspore.ops.ArgMinWithValue.html#mindspore.ops.ArgMinWithValue).

## Differences

PyTorch: Output tuple(min, index of min).

MindSpore: Output tuple(index of min, min).

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# Output tuple(index of min, min).
input_x = ms.Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), ms.float32)
argmin = ops.ArgMinWithValue()
index, output = argmin(input_x)
print(index)
print(output)
# Out：
# 0
# 0.0

# Output tuple(min, index of min).
input_x = torch.tensor([0.0, 0.4, 0.6, 0.7, 0.1])
output, index = torch.min(input_x, 0)
print(index)
print(output)
# Out：
# tensor(0)
# tensor(0.)
```
