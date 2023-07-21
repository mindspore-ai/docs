# Differences between torch.nn.Dropout and mindspore.nn.Dropout

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Dropout.md)

## torch.nn.Dropout

```python
class torch.nn.Dropout(
    p=0.5,
    inplace=False
)
```

For more information, see[torch.nn.Dropout](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Dropout).

## mindspore.nn.Dropout

```python
class mindspore.nn.Dropout(
    keep_prob=0.5,
    dtype=mstype.float
)
```

For more information, see[mindspore.nn.Dropout](https://mindspore.cn/docs/en/r1.9/api_python/nn/mindspore.nn.Dropout.html#mindspore.nn.Dropout).

## Use Pattern

PyTorch：**p** - Probability of an element to be zeroed. Default: 0.5.

PyTorch: The parameter P is the probability of discarding the parameter.

MindSpore：**keep_prob** ([*float*](https://docs.python.org/library/functions.html#float)) - The keep rate, greater than 0 and less equal than 1. E.g. rate=0.9, dropping out 10% of input units. Default: 0.5.

MindSpore：The parameter keep_prob is the probability of carding the parameter.

## Code Example

```python
# The following implements Dropout with MindSpore.
import torch.nn
import mindspore.nn
import numpy as np

m = torch.nn.Dropout(p=0.9)
input = torch.tensor(np.ones([5,5]),dtype=float)
output = m(input)
print(tuple(output.shape))
# out:
# (5, 5)

input = mindspore.Tensor(np.ones([5,5]),mindspore.float32)
net = mindspore.nn.Dropout(keep_prob=0.1)
net.set_train()
output = net(input)
print(output.shape)
# out:
# (5, 5)
```
