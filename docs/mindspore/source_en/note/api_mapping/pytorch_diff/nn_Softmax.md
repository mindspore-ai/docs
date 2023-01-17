# Function Differences with torch.nn.Softmax

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/nn_Softmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Softmax

```text
class torch.nn.Softmax(dim=None)(input) -> Tensor
```

For more information, see [torch.nn.Softmax](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmax.html.

## mindspore.nn.Softmax

```text
class mindspore.nn.Softmax(axis=-1)(x) -> Tensor
```

For more information, see [mindspore.nn.Softmax](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Softmax.html).

## Differences

PyTorch: a binary classification function, a generalization on multiclassification, which aims to present the results of multiclassification in the form of probabilities.

MindSpore: MindSpore API implements the same function as PyTorch, but the parameter names are different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | dim     | axis      | Same function, different parameter names |
|      | Parameter 2 | input  | x   | Same function, different parameter names |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import numpy
from torch import tensor
import torch.nn as nn

x = torch.FloatTensor([1, 1])
softmax = nn.Softmax(dim=0)(x)
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

### Code Example 2

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import numpy
from torch import tensor
import torch.nn as nn

x = torch.FloatTensor([1, 1, 1, 1])
softmax = nn.Softmax(dim=0)(x)
print(softmax.numpy())
# [0.25 0.25 0.25 0.25]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.array([1, 1, 1, 1]), mindspore.float16)
softmax = mindspore.nn.Softmax()
output = softmax(x)
print(output)
# [0.25 0.25 0.25 0.25]
```