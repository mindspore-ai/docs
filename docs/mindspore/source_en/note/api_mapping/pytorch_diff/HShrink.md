# Function Differences with torch.nn.Hardshrink

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/HShrink.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Hardshrink

```text
torch.nn.Hardshrink(lambd=0.5)(input) -> Tensor
```

For more information, see [torch.nn.Hardshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardshrink.html#torch.nn.Hardshrink).

## mindspore.nn.HShrink

```text
mindspore.nn.HShrink(lambd=0.5)(input_x) -> Tensor
```

For more information, see [mindspore.nn.HShrink](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.HShrink.html).

## Differences

PyTorch: Activation function, and calculate the output by the input elements.

MindSpore: MindSpore API implements the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | lambd   | lambd     | -    |
|      | Parameter 2 | input   | input_x     | Same function, different parameter names    |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.Hardshrink()
input = torch.tensor([[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]], dtype=torch.float32)
output = m(input)
output = output.detach().numpy()
print(output)
# [[ 0.      1.      2.    ]
#  [ 0.      0.     -2.1233]]

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

input_x = Tensor(np.array([[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]), mindspore.float32)
hshrink = nn.HShrink()
output = hshrink(input_x)
print(output)
# [[ 0.      1.      2.    ]
#  [ 0.      0.     -2.1233]]
```

