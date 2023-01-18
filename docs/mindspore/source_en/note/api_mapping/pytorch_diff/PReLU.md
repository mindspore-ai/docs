# Function Differences with torch.nn.PReLU

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/PReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.PReLU

```text
class torch.nn.PReLU(num_parameters=1, init=0.25)(input) -> Tensor
```

For more information, see [torch.nn.PReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.PReLU.html).

## mindspore.nn.PReLU

```text
class mindspore.nn.PReLU(channel=1, w=0.25)(x) -> Tensor
```

For more information, see [mindspore.nn.PReLU](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.PReLU.html).

## Differences

PyTorch: PReLU activation function.

MindSpore: MindSpore implements the same function as PyTorch, but with different parameter names.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | num_parameters | channel | Same function, different parameter names |
| | Parameter 2 | init | w | Same function, different parameter names |
| Input | Single input | input | x | Same function, different parameter names |

### Code Example 1

> This function is the same for both APIs, same usage and same default value. Only the parameter names are different.

```python
# PyTorch
import torch
from torch import tensor
from torch import nn
import numpy as np

x = tensor(np.array([[0.1, -0.6], [-0.9, 0.9]]), dtype=torch.float32)
m = nn.PReLU()
out = m(x)
output = out.detach().numpy()
print(output)
# [[ 0.1   -0.15 ]
#  [-0.225  0.9  ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x = Tensor(np.array([[0.1, -0.6], [-0.9, 0.9]]), mindspore.float32)
prelu = nn.PReLU()
output = prelu(x)
print(output)
# [[ 0.1   -0.15 ]
#  [-0.225  0.9  ]]
```

### Code Example 2

> If do not use the default value, you can use MindSpore to achieve the same function by simply setting the corresponding parameter to an equal number.

```python
# PyTorch
import torch
from torch import tensor
from torch import nn
import numpy as np

x = tensor(np.array([[0.1, -0.6], [-0.5, 0.9]]), dtype=torch.float32)
m = nn.PReLU(num_parameters=1, init=0.5)
out = m(x)
output = out.detach().numpy()
print(output)
# [[ 0.1  -0.3 ]
#  [-0.25  0.9 ]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([[0.1, -0.6], [-0.5, 0.9]]), mindspore.float32)
prelu = nn.PReLU(channel=1, w=0.5)
output = prelu(x)
print(output)
# [[ 0.1  -0.3 ]
#  [-0.25  0.9 ]]
```
