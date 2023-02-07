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
|      | Parameter 6 |     -     | label_smoothing | Label smoothing value, used as a regularization means to prevent overfitting of the model when calculating Loss. The range of values is [0.0, 1.0]. Default value: 0.0. |
| Input | Input 1 | input    | logits | Same function, different parameter names    |
|      | Input 2| target   | labels  | Same function, different parameter names   |

### Code Example 1

> Both PyTorch and MindSpore support the case where the target value is a scalar.

```python
# PyTorch
import torch
import numpy as np

input_torch = np.array([[1.624, -0.611, -0.528, -1.072, 0.865], [-2.301, 1.744, -0.761, 0.319, -0.249], [1.462, -2.060, -0.322, -0.384, 1.133]])
target_torch = np.array([1, 0, 4])
loss = torch.nn.CrossEntropyLoss()
input_torch = torch.tensor(input_torch, requires_grad=True)
target_torch = torch.tensor(target_torch, dtype=torch.long)
output = loss(input_torch, target_torch)
print(round(float(output.detach().numpy()), 3))
# 2.764

# MindSpore
import mindspore
import numpy as np

input_ms = np.array([[1.624, -0.611, -0.528, -1.072, 0.865], [-2.301, 1.744, -0.761, 0.319, -0.249], [1.462, -2.060, -0.322, -0.384, 1.133]])
target_ms = np.array([1, 0, 4])
input_ms = mindspore.Tensor(input_ms, mindspore.float32)
target_ms = mindspore.Tensor(target_ms, mindspore.int32)
loss = mindspore.nn.CrossEntropyLoss()
output = loss(input_ms, target_ms)
print(round(float(output), 3))
# 2.764
```

