# Function Differences with torch.max

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/max.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torch.max

```python
torch.max(
    input,
    dim,
    keepdim=False,
    out=None
)
```

For more information, see [torch.max](https://pytorch.org/docs/1.8.1/torch.html#torch.max).

## mindspore.ops.max

```python
class mindspore.ops.max(
    x,
    axis=0,
    keep_dims=False
)
```

For more information, see [mindspore.ops.max](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.max.html).

## Differences

PyTorch: Output tuple(max, index of max).

MindSpore: Output tuple(index of max, max).

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | input        | x       | Same function, different parameter names |
|      | Parameter 2 | dim       | axis      | Same function, different parameter names |
|      | Parameter 3 | keepdim |  keep_dims   | Same function, different parameter names|
| | Parameter 4 | out | - | Not involved |

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# Output tuple(index of max, max).
input_x = ms.Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), ms.float32)
index, output = ops.max(input_x)
print(index)
# 3
print(output)
# 0.7

# Output tuple(max, index of max).
input_x = torch.tensor([0.0, 0.4, 0.6, 0.7, 0.1])
output, index = torch.max(input_x, 0)
print(index)
# tensor(3)
print(output)
# tensor(0.7000)
```
