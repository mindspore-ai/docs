# Differences between torch.nn.Dropout and mindspore.nn.Dropout

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mindspore.nn.Dropout.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see[mindspore.nn.Dropout](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Dropout.html#mindspore.nn.Dropout).

## Use Pattern

The probability of P parameter in PyTorch is the dropping parameter.

The probability of keep_prob parameter in MindSpore is the retention parameter.

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
