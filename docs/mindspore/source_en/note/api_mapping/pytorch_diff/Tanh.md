# Function Differences with torch.nn.Tanh

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Tanh.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Tanh

```text
class torch.nn.Tanh()(input) -> Tensor
```

For more information, see [torch.nn.Tanh](https://pytorch.org/docs/1.8.1/generated/torch.nn.Tanh.html).

## mindspore.nn.Tanh

```text
class mindspore.nn.Tanh()(x) -> Tensor
```

For more information, see [mindspore.nn.Tanh](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Tanh.html).

## Differences

PyTorch: Compute the hyperbolic tangent function tanh.

MindSpore: MindSpore API implements the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Input | Single input | input      | x         | Same function, different parameter names  |

### Code Example

> Compute the tanh function for input `x`, and MindSpore API function is consistent with PyTorch.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor, nn

m = nn.Tanh()
x_ = np.array([0.7713, 0.0208, 0.6336], dtype=np.float32)
x = tensor(x_)
output = m(x)
print(output.numpy())
# [0.64768475 0.020797   0.56052613]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, nn

m = nn.Tanh()
x_ = np.array([0.7713, 0.0208, 0.6336], dtype=np.float32)
x = Tensor(x_, mindspore.float32)
output = m(x)
print(output)
# [0.64768475 0.020797   0.56052613]
```
