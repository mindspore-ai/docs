# Function Differences with torch.erfinv

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/erfinv.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.erfinv

```text
torch.erfinv(input, *, out=None) -> Tensor
```

For more information, see [torch.erfinv](https://pytorch.org/docs/1.8.1/generated/torch.erfinv.html).

## mindspore.ops.erfinv

```text
mindspore.ops.erfinv(input) -> Tensor
```

For more information, see [mindspore.ops.erfinv](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.erfinv.html).

## Differences

PyTorch: Compute the inverse error function of the input Tensor. The inverse error function is in the range (-1, 1) with the formula: $erfinv(erf(x))=x$.

MindSpore: MindSpore API implements the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | input | input | - |
| | Parameter 2 | out | - |Not involved |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

input_x = tensor([0, 0.5, -0.9], dtype=torch.float32)
out = torch.erfinv(input_x).numpy()
print(out)
# [ 0.          0.47693628 -1.1630871 ]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([0, 0.5, -0.9]), mindspore.float32)
output = mindspore.ops.erfinv(x)
print(output)
# [ 0.          0.47693628 -1.163087  ]
```
