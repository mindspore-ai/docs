# Function Differences with torch.cosh

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cosh.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.cosh

```text
torch.cosh(input, *, out=None) -> Tensor
```

For more information, see [torch.cosh](https://pytorch.org/docs/1.8.1/generated/torch.cosh.html).

## mindspore.ops.cosh

```text
mindspore.ops.cosh(x) -> Tensor
```

For more information, see [mindspore.ops.cosh](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cosh.html).

## Differences

PyTorch: Return a new tensor with the hyperbolic cosine of the input element.

MindSpore: Same function, and only the parameter names are different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | input   | x         | Same function, different parameter names                     |
|      | Parameter 2 | out     | -         | Not involved |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# Example 1 -> Code example 1
# PyTorch
import torch

a = torch.tensor([0.24, 0.83, 0.31, 0.09],dtype=torch.float32)
output = torch.cosh(a)
output = output.detach().numpy()
print(output)
# [1.0289385 1.364684  1.048436  1.0040528]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
output = ops.cosh(x)
print(output)
# [1.0289385 1.364684  1.048436  1.0040528]
```
