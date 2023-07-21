# Differences with torch.nn.Softmax

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/nn_Softmax.md)

## torch.nn.Softmax

```text
class torch.nn.Softmax(dim=None)(input) -> Tensor
```

For more information, see [torch.nn.Softmax](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmax.html.

## mindspore.nn.Softmax

```text
class mindspore.nn.Softmax(axis=-1)(x) -> Tensor
```

For more information, see [mindspore.nn.Softmax](https://www.mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.Softmax.html).

## Differences

PyTorch: a generalization of the binary classification function on multiclassification, which aims to present the results of multiclassification in the form of probabilities.

MindSpore: MindSpore API implements the same function as PyTorch, but the parameter names are different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | dim     | axis      | Same function, different parameter names, different default values |
|   Input   | Single input | input  | x   | Same function, different parameter names |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import numpy
from torch import tensor
import torch.nn as nn

x = torch.FloatTensor([1, 1])
softmax = nn.Softmax(dim=-1)(x)
print(softmax.numpy())
# [0.5 0.5]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.array([1, 1]), mindspore.float16)
softmax = mindspore.nn.Softmax()
output = softmax(x)
print(output)
# [0.5 0.5]
```
