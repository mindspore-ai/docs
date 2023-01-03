# Function Differences with torch.not_equal

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/not_equal.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.not_equal

```text
torch.not_equal(input, other, *, out=None) -> Tensor
```

For more information, see [torch.not_equal](https://pytorch.org/docs/1.8.1/generated/torch.not_equal.html).

## mindspore.ops.not_equal

```text
mindspore.ops.not_equal(x, other) -> Tensor
```

For more information, see [mindspore.ops.not_equal](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/ops/mindspore.ops.not_equal.html).

## Differences

PyTorch: Computes whether `input` and `other` are not equal, element-wise.

MindSpore: MindSpore API implements the same functionality as PyTorch except for the `input` parameter name.

| Categories | Subcategories | PyTorch | MindSpore | Differences       |
| --- |---------------|---------|-----------|-------------------------------------------------------------------|
|Parameter | Parameter 1   | input   | x         | The function is the same, and the parameter `input` name is different |
| | Parameter 2   | other   | other     | The function is the same          |
| | Parameter 3   | out     | -         | MindSpore does not have this Parameter      |

## Code Example 1

> The two APIs have the same functions and the same usage.

```python
# PyTorch
import torch
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
output = torch.not_equal(input, other)
print(output.detach().numpy())
#[[False  True]
# [ True False]]

# MindSpore
import mindspore
from mindspore import Tensor, ops
import numpy as np
x = Tensor(np.array([[1, 2], [3, 4]]))
other = Tensor(np.array([[1, 1], [4, 4]]))
output = ops.not_equal(x, other)
print(output)
#[[False  True]
# [ True False]]
```
