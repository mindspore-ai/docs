# Function Differences with torch.nn.functional.elu

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/elu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.elu

```text
torch.nn.functional.elu(input, alpha=1.0, inplace=False) -> Tensor
```

For more information, see [torch.nn.functional.elu](https://pytorch.org/docs/1.8.1/nn.functional.html#elu).

## mindspore.ops.elu

```text
mindspore.ops.elu(input_x, alpha=1.0) -> Tensor
```

For more information, see [mindspore.ops.elu](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.elu.html).

## Differences

PyTorch: compute the exponential linear value of the input x. The result is $ \text{elu}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1)) $, with the inplace parameter to choose whether to operate in-place or not, and the default is False.

MindSpore: MindSpore API implements the same functions as PyTorch, but α currently only supports 1.0.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ----|
|Parameters | Parameter 1 | input | input_x |Same function, different parameter names |
| | Parameter 2 | alpha | alpha | α factor. MindSpore currently only supports alpha equal to 1.0 |
| | Parameter 3 | inplace | - | MindSpore does not have this parameter |

### Code Example 1

> Both APIs implement the exponential linear unit function, but PyTorch can customize the α coefficient, and MindSpore only supports a coefficient of 1.0.

```python
# PyTorch
import torch
from torch import tensor
from torch.nn.functional import elu
import numpy as np

x_ = np.array([[np.arange(-6,0).reshape(2, 3),np.arange(0,6).reshape(2, 3)]])
x = tensor(x_, dtype=torch.float32)
output = elu(x, alpha = 1).detach().numpy()
print(output)
# [[[[-0.9975212  -0.99326205 -0.9816844 ]
#   [-0.95021296 -0.86466473 -0.63212055]]
#
#  [[ 0.          1.          2.        ]
#   [ 3.          4.          5.        ]]]]

# MindSpore
import mindspore as ms
from mindspore import ops
import numpy as np

x_ = np.array([[np.arange(-6,0).reshape(2, 3),np.arange(0,6).reshape(2, 3)]])
x = ms.Tensor(x_, ms.float32)
output = ops.elu(x)
print(output)
# [[[[-0.9975212  -0.99326205 -0.9816844 ]
#    [-0.95021296 -0.86466473 -0.6321205 ]]
#
#   [[ 0.          1.          2.        ]
#    [ 3.          4.          5.        ]]]]
```
