# Function Differences with torch.conj

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/conj.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.conj

```text
torch.conj(input, *, out=None) -> Tensor
```

For more information, see [torch.conj](https://pytorch.org/docs/1.8.1/generated/torch.conj.html).

## mindspore.ops.conj

```text
mindspore.ops.conj(input) -> Tensor
```

For more information, see [mindspore.ops.conj](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.conj.html).

## Differences

PyTorch: Return the conjugate complex of the input tensor.

MindSpore: MindSpore API implements the same function as PyTorch.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input   | input     | -    |
|      | Parameter 2 | out     | -         | Not involved        |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

a = tensor([-1 + 1j, -2 + 2j, 3 - 3j], dtype=torch.complex64)
b = torch.conj(a)
print(b.detach().numpy())
# [-1.-1.j -2.-2.j 3.+3.j]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

a = Tensor([-1 + 1j, -2 + 2j, 3 - 3j], dtype=mindspore.complex64)
b = ops.conj(a)
print(b)
# [-1.-1.j -2.-2.j 3.+3.j]
```
