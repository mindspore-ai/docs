# 比较与torch.nn.TripletMarginLoss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TripletMarginLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.nn.TripletMarginLoss    |   mindspore.nn.TripletMarginLoss   |
|    torch.functional.triplet_margin_loss   |  mindspore.ops.triplet_margin_loss   |

## torch.nn.TripletMarginLoss

```text
torch.nn.TripletMarginLoss(
    margin=1.0,
    p=2.0,
    eps=1e-06,
    swap=False,
    size_average=None,
    reduce=None,
    reduction='mean'
)(anchor, positive, negative) -> Tensor/Scalar
```

更多内容详见[torch.nn.TripletMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.TripletMarginLoss.html)。

## mindspore.nn.TripletMarginLoss

```text
mindspore.nn.TripletMarginLoss(
    p=2,
    swap=False,
    eps=1e-06,
    reduction='mean'
)(margin, x, positive, negative) -> Tensor/Scalar
```

更多内容详见[mindspore.nn.TripletMarginLoss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.TripletMarginLoss.html)。

## 差异对比

PyTorch：

- PyTorch的接口有两个初始化参数 `size_average` 和 `reduce` 。这两个参数已被弃用，并使用 `reduction` 取代。

- `margin` 是一个初始化参数，而不是输入参数。其数据类型是float。

MindSpore：

- PyTorch的接口没有初始化参数 `size_average` 和 `reduce` 。

- `margin` 不是初始化参数，而是输入参数。其数据类型可以是tensor或float。

- MindSpore的输入参数 `x` 对应PyTorch的输入参数 `anchor` 。

- MindSpore的初始化参数 `swap` 和 `eps` 位置顺序和PyTorch不同。

功能上无差异。

| 分类       | 子类         | PyTorch      | MindSpore   | 差异          |
| ---------- | ------------ | ------------ | ---------   | ------------- |
| 参数       | 参数 1       | margin       | -           | 位置不同，功能一致。数据类型是float。 |
|            | 参数 2       | p            | p           | -             |
|            | 参数 3       | eps          | swap        | 位置不同，功能一致。 |
|            | 参数 4       | swap         | eps         | 位置不同，功能一致。 |
|            | 参数 5       | size_average | -           | PyTorch已弃用该参数，MindSpore没有该参数。 |
|            | 参数 6       | reduce       | -           | PyTorch已弃用该参数，MindSpore没有该参数。 |
|            | 参数 7       | reduction    | reduction   | -             |
| 输入       | 输入 1       | -            | margin      | 位置不同，功能一致。数据类型是tensor或float。 |
|            | 输入 2       | anchor       | x           | 功能一致，参数名不同。 |
|            | 输入 3       | positive     | positive    | -             |
|            | 输入 4       | negative     | negative    | -             |

## 差异分析与示例

```python
# PyTorch
import torch
import torch.nn as nn
import numpy as np

p = 2
swap = False
eps = 1e-06
reduction = 'mean'
margin = 1.0
triplet_margin_loss = nn.TripletMarginLoss(margin=margin, p=p, eps=eps, swap=swap, reduction=reduction)

x = torch.tensor(np.array([[0.3, 0.7], [0.5, 0.5]]), dtype=torch.float32)
positive = torch.tensor(np.array([[0.4, 0.6], [0.4, 0.6]]), dtype=torch.float32)
negative = torch.tensor(np.array([[0.2, 0.9], [0.3, 0.7]]), dtype=torch.float32)
output = triplet_margin_loss(x, positive, negative)
print(output)
# tensor(0.8882)

# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

p = 2
swap = False
eps = 1e-06
reduction = 'mean'
triplet_margin_loss = nn.TripletMarginLoss(p=p, swap=swap, eps=eps, reduction=reduction)

x = ms.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]), dtype=ms.float32)
positive = ms.Tensor(np.array([[0.4, 0.6], [0.4, 0.6]]), dtype=ms.float32)
negative = ms.Tensor(np.array([[0.2, 0.9], [0.3, 0.7]]), dtype=ms.float32)
margin = ms.Tensor(1.0, ms.float32)
output = triplet_margin_loss(x, positive, negative, margin)
print(output)
# 0.8881968
```
