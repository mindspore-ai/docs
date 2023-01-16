# Function Differences with torch.nn.functional.softmax

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/softmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.softmax

```python
torch.nn.functional.softmax(
    input,
    dim=None,
    _stacklevel=3,
    dtype=None
)
```

For more information, see [torch.nn.functional.softmax](https://pytorch.org/docs/1.8.0/nn.functional.html#torch.nn.functional.softmax).

## mindspore.ops.softmax

```python
class mindspore.ops.softmax(
    x,
    axis=-1,
)
```

For more information, see [mindspore.ops.softmax](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.softmax.html#mindspore.ops.softmax).

## Differences

PyTorch: Supports to implement the function with the `dim` parameter and `input`, scaling the specified dimension elements between [0, 1] and the total to 1.

MindSpore: Supports to implement the function with the `axis` parameter and `x`, scaling the specified dimension elements between [0, 1] and the total to 1.

| Classification | Subclass  | PyTorch | MindSpore | difference |
| ---- | ----- | ------- | --------- | -------------------- |
| parameter | parameter1 | input   | x   | Same functions, different parameter names |
|      | parameter2 | dim     | axis   | Same functions, different parameter names |
|      | parameter3 | dtype    | -   | Uesd to specify the type of data for the output Tensor in PyTorch. This parameter does not exist in MindSpore |

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, we can define an instance of this class first, and the default value of the parameter axis is -1.
logits = ms.Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
output1 = ops.softmax(logits)
print(output1)
# Out:
# [0.01165623 0.03168492 0.08612854 0.23412167 0.6364086 ]
logits = ms.Tensor(np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), ms.float32)
output2 = ops.softmax(logits, axis=0)
print(output2)
# out:
# [[0.01798621 0.11920292 0.5        0.880797   0.98201376]
#  [0.98201376 0.880797   0.5        0.11920292 0.01798621]]

# In torch, the input and dim should be input at the same time to implement the function.
input = torch.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
output3 = torch.nn.functional.softmax(input, dim=0)
print(output3)
# Out:
# tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364], dtype=torch.float64)
```
