# Function Differences with torch.nn.TripletMarginLoss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TripletMarginLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For the functional differences between `torch.nn.functional.triplet_margin_loss` and `mindspore.ops.triplet_margin_loss` , refer to the functional differences between `torch.nn.TripletMarginLoss` and `mindspore.nn.TripletMarginLoss` .

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

For more information, see [torch.nn.TripletMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.TripletMarginLoss.html).

## mindspore.nn.TripletMarginLoss

```text
mindspore.nn.TripletMarginLoss(
    p=2,
    swap=False,
    eps=1e-06,
    reduction='mean'
)(margin, x, positive, negative) -> Tensor/Scalar
```

For more information, see [mindspore.nn.TripletMarginLoss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.TripletMarginLoss.html).

## Differences

PyTorch:

- PyTorch has two initialization parameters `size_average` and `reduce` , which are deprecated and replaced by `reduction` .

- `margin` is an initialization parameter, not an input parameter. The data type of `margin` is float.

MindSpore:

- MindSpore doesn't have initialization parameters `size_average` and `reduce` .

- `margin` is not an initialization parameter, but an input parameter. The data type of `margin` can be tensor or float.

- The input parameter `x` of MindSpore corresponds to the input parameter `anchor` of PyTorch.

- MindSpore's initialization parameters 'swap' and 'eps' are positioned in a different order than PyTorch.

| Categories | Subcategories | PyTorch      | MindSpore   | Differences   |
| ---------- | ------------- | ------------ | ---------   | ------------- |
| Parameters | Parameter 1   | margin       | -           | Different position, same function. Data type is float. |
|            | Parameter 2   | p            | p           | -             |
|            | Parameter 3   | eps          | swap        | Different position, same function. |
|            | Parameter 4   | swap         | eps         | Different position, same function. |
|            | Parameter 5   | size_average | -           | PyTorch has deprecated this parameter, while MindSpore does not have this parameter. |
|            | Parameter 6   | reduce       | -           | PyTorch has deprecated this parameter, while MindSpore does not have this parameter. |
|            | Parameter 7   | reduction    | reduction   | -             |
| Input      | Input 1       | -            | margin      | Different position, same function. Data type can be tensor or float. |
|            | Input 2       | anchor       | x           | Same function, different parameter names. |
|            | Input 3       | positive     | positive    | -             |
|            | Input 4       | negative     | negative    | -             |

### Code Example

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
triplet_margin_loss = nn.TripletMarginLoss(p=p, eps=eps, swap=swap, reduction=reduction)

x = ms.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]), dtype=ms.float32)
positive = ms.Tensor(np.array([[0.4, 0.6], [0.4, 0.6]]), dtype=ms.float32)
negative = ms.Tensor(np.array([[0.2, 0.9], [0.3, 0.7]]), dtype=ms.float32)
margin = ms.Tensor(1.0, ms.float32)
output = triplet_margin_loss(x, positive, negative, margin)
print(output)
# 0.8881968
```
