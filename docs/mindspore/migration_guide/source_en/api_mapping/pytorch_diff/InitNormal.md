# Function Differences with torch.nn.init.normal_

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/InitNormal.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torch.nn.init.normal_

```python
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
```

For more information, see [torch.nn.init.normal_](https://pytorch.org/docs/1.5.0/nn.init.html#torch.nn.init.normal_).

## mindspore.common.initializer.Normal

```python
mindspore.common.initializer.Normal(sigma=0.01, mean=0.0)
```

For more information, see [mindspore.common.initializer.Normal](https://mindspore.cn/docs/api/en/r1.5/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Normal).

## Differences

PyTorch: Obtain values N(std, mean) from the normal distribution. Default: std=1., mean=0.0.

MindSpore: Obtain values N(sigma, mean) from the normal distribution. Default: sigma=0.01, mean=0.0.

## Code Example

> The following code will generate random results.

```python
import mindspore
from mindspore.common.initializer import Normal, initializer

w = initializer(Normal(sigma=1, mean=0.0), shape=[3, 4], dtype=mindspore.float32)
print(w)

# out
# [[ 1.154151   -2.0898762  -0.652796    1.4034489 ]
# [-1.415637    1.717648   -0.6167477  -1.2566634 ]
# [ 3.330741    0.49453223  1.9247946  -0.49406782]]

import torch
from torch import nn

w = nn.init.normal_(torch.empty(3, 4), mean=0., std=1.)
print(w)
# out
# tensor([[ 0.0305, -1.1593,  1.0516, -1.0172],
#         [-0.1539,  0.0793,  0.9397, -0.1186],
#         [ 2.6214,  0.5601,  0.7149, -0.4375]])
```
