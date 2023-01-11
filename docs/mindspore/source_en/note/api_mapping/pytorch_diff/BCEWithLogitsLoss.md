# Function Differences with torch.nn.BCEWithLogitsLoss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BCEWithLogitsLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.BCEWithLogitsLoss

```text
torch.nn.BCEWithLogitsLoss(
    weight=None,
    size_average=None,
    reduce=None,
    reduction='mean',
    pos_weight=None
)(input, target) -> Tensor
```

For more information, see [torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.BCEWithLogitsLoss.html).

## mindspore.nn.BCEWithLogitsLoss

```text
class mindspore.nn.BCEWithLogitsLoss(
    reduction='mean',
    weight=None,
    pos_weight=None
)(logits, labels) -> Tensor
```

For more information, see [mindspore.nn.BCEWithLogitsLoss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BCEWithLogitsLoss.html).

## Differences

PyTorch: Combine the Sigmoid layer and BCELoss in one class to calculate the binary cross-entropy loss between the predicted and target values, making it numerically more stable than using Sigmoid followed by BCELoss separately.

MindSpore: MindSpore API basically implements the same function as PyTorch. Only the input parameter names are different.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input | logits | input Tensor |
| | Input 2 | target | labels | input Tensor |
| Parameters | Parameter 1 | weight | weight | Same function, same parameter name |
| | Parameter 2 | size_average | - | Same function. PyTorch has deprecated this parameter, while MindSpore does not have this parameter |
| | Parameter 3 | reduce | - | Same function. PyTorch has deprecated this parameter, while MindSpore does not have this parameter |
| | Parameter 4 | reduction | reduction | Same function, same parameter name |
| | Parameter 5 | pos_weight | pos_weight | Same function, same parameter name |

### Code Example 1

> The two APIs achieve the same function and have the same usage. The three parameters of PyTorch BCEWithLogitsLoss operator, weight, reduction, and pos_weight, are functionally identical to the corresponding three parameters of MindSpore BCEWithLogitsLoss operator, with the same parameter names and the same default values. By default, MindSpore can get the same results as PyTorch.

```python
# PyTorch
import torch
from torch import Tensor
import numpy as np

np.random.seed(1)
input = Tensor(np.random.rand(1, 2, 3).astype(np.float32))
print(input.numpy())
# [[[4.17021990e-01 7.20324516e-01 1.14374816e-04]
#   [3.02332580e-01 1.46755889e-01 9.23385918e-02]]]
target = Tensor(np.random.randint(2,size=(1, 2, 3)).astype(np.float32))
print(target.numpy())
# [[[0. 1. 1.]
#   [0. 0. 1.]]]
torch_BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
torch_output = torch_BCEWithLogitsLoss(input, target)
torch_output_np = torch_output.numpy()
print(torch_output_np)
# 0.7142954

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

np.random.seed(1)
logits = Tensor(np.random.rand(1, 2, 3).astype(np.float32))
print(logits.asnumpy())
# [[[4.17021990e-01 7.20324516e-01 1.14374816e-04]
#   [3.02332580e-01 1.46755889e-01 9.23385918e-02]]]
labels = Tensor(np.random.randint(2,size=(1, 2, 3)).astype(np.float32))
print(labels.asnumpy())
# [[[0. 1. 1.]
#   [0. 0. 1.]]]
ms_BCEWithLogitsLoss = mindspore.nn.BCEWithLogitsLoss()
ms_output = ms_BCEWithLogitsLoss(logits, labels)
ms_output_np = ms_output.asnumpy()
print(ms_output_np)
# 0.71429545
```
