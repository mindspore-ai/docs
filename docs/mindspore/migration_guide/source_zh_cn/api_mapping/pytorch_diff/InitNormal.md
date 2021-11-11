# 比较与torch.nn.init.normal_的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/InitNormal.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.init.normal_

```python
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
```

更多内容详见[torch.nn.init.normal_](https://pytorch.org/docs/1.5.0/nn.init.html#torch.nn.init.normal_)。

## mindspore.common.initializer.Normal

```python
mindspore.common.initializer.Normal(sigma=0.01, mean=0.0)
```

更多内容详见[mindspore.common.initializer.Normal](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Normal)。

## 使用方式

PyTorch：默认输出均值为0，标准差为1的正态分布。使用时传入均值和标准差。

MindSpore：默认均值为0，标准差为0.01的正态分布。使用时传入均值和标准差。

## 代码示例

> 下述代码结果有随机性。

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
