# Function Differences with torch.nn.CrossEntropyLoss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CrossEntropyLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.CrossEntropyLoss

```text
class torch.nn.CrossEntropyLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean'
)(input, target) -> Tensor
```

For more information, see [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CrossEntropyLoss.html).

## mindspore.nn.CrossEntropyLoss

```text
class mindspore.nn.CrossEntropyLoss(
    weight=None,
    ignore_index=-100,
    reduction='mean',
    label_smoothing=0.0
)(logits, labels) -> Tensor
```

For more information, see [mindspore.nn.CrossEntropyLoss](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.CrossEntropyLoss.html).

## Differences

PyTorch: Calculate the cross-entropy loss between the predicted and target values.

MindSpore: MindSpore implements the same function as PyTorch, and the target value supports two different data forms: scalar and probabilistic.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | weight | weight      | - |
|  | Parameter 2 | size_average | - | Deprecated, function taken over by reduction |
|      | Parameter 3 | ignore_index | ignore_index    | - |
|      | Parameter 4 | reduce | - | Deprecated, function taken over by reduction |
|      | Parameter 5 | reduction    | reduction       | - |
|      | Parameter 6 | input    | logits    | Same function, different parameter names  |
|      | Parameter 7 | target    | labels   | Same function, different parameter names |
|      | Parameter 8 |    -     | label_smoothing | Label smoothing value, used as a regularization means to prevent overfitting of the model when calculating Loss. The range of values is [0.0, 1.0]. Default value: 0.0. |

### Code Example 1

> Both PyTorch and MindSpore support the case where the target value is a scalar.

```python
# PyTorch
import torch
import numpy as np

inpu = np.array([[1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763], [-2.3015387, 1.74481176, -0.7612069, 0.3190391, -0.24937038], [1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944]])
targe = np.array([1, 0, 4])

loss = torch.nn.CrossEntropyLoss()
input = torch.tensor(inpu, requires_grad=True)
target = torch.tensor(targe, dtype=torch.long)
output = loss(input, target)
print(output.detach().numpy())
# 2.7648239812294704

# MindSpore
import mindspore
import numpy as np

inputs = mindspore.Tensor(inpu, mindspore.float32)
target = mindspore.Tensor(targe, mindspore.int32)
loss = mindspore.nn.CrossEntropyLoss()
output = loss(inputs, target)
print(output)
# 2.7648222
```

