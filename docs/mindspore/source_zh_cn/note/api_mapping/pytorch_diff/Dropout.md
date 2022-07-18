# 比较torch.nn.Dropout与mindspore.nn.Dropout的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Dropout.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Dropout

```python
class torch.nn.BatchNorm1d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)
class torch.nn.Dropout(
    p=0.5,
    inplace=False
)
```

更多内容详见[torch.nn.Dropout](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Dropout)。

## mindspore.nn.Dropout

```python
class mindspore.nn.Dropout(
    keep_prob=0.5,
    dtype=mstype.float
)
```

更多内容详见[mindspore.nn.Dropout](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Dropout.html#mindspore.nn.Dropout)。

## 使用方式

PyTorch中P参数为丢弃参数的概率。

MindSpore中keep_prob参数为保留参数的概率。

## 代码示例

```python
# The following implements Dropout with MindSpore.
import torch.nn
import mindspore.nn
import numpy as np

m = torch.nn.Dropout(p=0.9)
input = torch.tensor(np.ones([5,5]),dtype=float)
output = m(input)
print(output)

# out:
#   [[0 10 0 0 0]
#   [0 0 0 0 0]
#   [0 0 10 0 0]
#   [0 10 0 0 0]
#   [0 0 0 0 10]]

input = mindspore.Tensor(np.ones([5,5]),mindspore.float32)
net = mindspore.nn.Dropout(keep_prob=0.1)
net.set_train()
output = net(input)
print(output)

# out:
#   [[0 10 0 0 0]
#   [0 0 0 10 0]
#   [0 0 0 0 0]
#   [0 10 10 0 0]
#   [0 0 10 0 0]]

```