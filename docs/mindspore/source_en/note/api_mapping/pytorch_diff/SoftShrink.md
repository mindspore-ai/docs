# Function Differences with torch.nn.SoftShrink

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SoftShrink.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Softshrink

```text
class torch.nn.Softshrink(lambd=0.5)(input) -> Tensor
```

For more information, see [torch.nn.Softshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softshrink.html).

## mindspore.nn.SoftShrink

```text
class mindspore.nn.SoftShrink(lambd=0.5)(input_x) -> Tensor
```

For more information, see [mindspore.nn.SoftShrink](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.SoftShrink.html).

## Differences

PyTorch: Used to calculate the Softshrink activation function.

MindSpore: The interface name is different from PyTorch. MindSpore is SoftShrink, while PyTorch is Softshrink, and the function is the same.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameter | Parameter 1 | lambd  | lambd     | - |
| Input | Single input | input  | input_x   | Same function, different parameter names |

### Code Example 1

> Compute the SoftShrink activation function for lambd=0.3.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor, nn

m = nn.Softshrink(lambd=0.3)
input_ = np.array([[0.5297, 0.7871, 1.1754], [0.7836, 0.6218, -1.1542]], dtype=np.float32)
input_t = tensor(input_)
output = m(input_t)
print(output.numpy())
# [[ 0.22969997  0.4871      0.8754    ]
#  [ 0.48359996  0.3218     -0.85419995]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, nn

m = nn.SoftShrink(lambd=0.3)
input_ = np.array([[0.5297, 0.7871, 1.1754], [0.7836, 0.6218, -1.1542]], dtype=np.float32)
input_t = Tensor(input_, mindspore.float32)
output = m(input_t)
print(output)
# [[ 0.22969997  0.4871      0.8754    ]
#  [ 0.48359996  0.3218     -0.85419995]]
```

### Code Example 2

> SoftShrink defaults to `lambd=0.5`.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor, nn

m = nn.Softshrink()
input_ = np.array([[0.5297, 0.7871, 1.1754], [0.7836, 0.6218, -1.1542]], dtype=np.float32)
input_t = tensor(input_)
output = m(input_t)
print(output.numpy())
# [[ 0.02969998  0.28710002  0.6754    ]
#  [ 0.28359997  0.12180001 -0.65419996]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, nn

m = nn.SoftShrink()
input_ = np.array([[0.5297, 0.7871, 1.1754], [0.7836, 0.6218, -1.1542]], dtype=np.float32)
input_t = Tensor(input_, mindspore.float32)
output = m(input_t)
print(output)
# [[ 0.02969998  0.28710002  0.6754    ]
#  [ 0.28359997  0.12180001 -0.65419996]]
```
