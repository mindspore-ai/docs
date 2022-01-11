# Function Differences with torch.nn.functional.softmax

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/Softmax.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.softmax

```python
torch.nn.functional.softmax(
    input,
    dim=None,
    _stacklevel=3,
    dtype=None
)
```

For more information, see [torch.nn.functional.softmax](https://pytorch.org/docs/1.5.0/nn.functional.html#torch.nn.functional.softmax).

## mindspore.ops.Softmax

```python
class mindspore.ops.Softmax(
    axis=-1,
)(logits)
```

For more information, see [mindspore.ops.Softmax](https://mindspore.cn/docs/api/en/r1.6/api_python/ops/mindspore.ops.Softmax.html#mindspore.ops.Softmax).

## Differences

PyTorch：Supports to implement the function with the `dim` parameter and input, scaling the specified dimension elements to between [0, 1] and the total to 1.

MindSpore：Supports to initialize the Softmax with the `axis` attribute, scaling the specified dimension elements to between [0, 1] and the total to 1.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, we can define an instance of this class first, and the default value of the parameter axis is -1.
logits = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
softmax = ops.Softmax()
output1 = softmax(logits)
print(output1)
# Out:
# [0.01165623 0.03168492 0.08612854 0.23412167 0.6364086 ]
logits = Tensor(np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), mindspore.float32)
softmax = ops.Softmax(axis=0)
output2 = softmax(logits)
print(output2)
# out:
# [[0.01798621 0.11920292 0.5        0.880797   0.98201376], [0.98201376 0.880797   0.5        0.11920292 0.01798621]]

# In torch, the input and dim should be input at the same time to implement the function.
input = torch.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
output3 = torch.nn.functional.softmax(input, dim=0)
print(output3)
# Out:
# tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364], dtype=torch.float64)

```
