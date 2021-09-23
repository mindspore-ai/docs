# 比较与torch.nn.init.uniform_的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/InitNormal.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.nn.init.uniform_

```python
torch.nn.init.uniform_(tensor, a=0., b=1.)
```

## mindspore.common.initializer.Uniform

```python
mindspore.common.initializer.Uniform(scale=0.07)
```

## 使用方式

PyTorch: 生成范围为 U(a, b) 的随机均匀分布。默认值：a=0, b=1.

MindSpore：生成范围为 U(-scale, scale) 的随机均匀分布。scale默认值为0.07。

## 代码示例

> 下述代码结果有随机性。

```python
import mindspore
from mindspore.common.initializer import Uniform, initializer

w = initializer(Uniform(1), shape=[3, 4], dtype=mindspore.float32)
print(w)
# out
# [[ 0.3710562  -0.8446909   0.07222586  0.4807771 ]
# [-0.4605304  -0.46153235  0.8576784   0.32084346]
# [ 0.97628933 -0.22949246 -0.3052143  -0.0164203 ]]

import torch
from torch import nn

w = nn.init.uniform_(torch.empty(3, 4), -1, 1)
print(w)
# out
# tensor([[ 0.2265,  0.2944,  0.4167, -0.4217],
#        [ 0.0971, -0.4190,  0.7143, -0.0494],
#        [ 0.4092, -0.1708, -0.9689,  0.3019]])
```
