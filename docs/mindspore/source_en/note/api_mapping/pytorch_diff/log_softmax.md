# Function Differences with torch.nn.functional.log_softmax

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/log_softmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.log_softmax

```python
torch.nn.functional.log_softmax(
    input,
    dim=None,
    _stacklevel=3,
    dtype=None
)
```

For more information, see [torch.nn.functional.log_softmax](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.log_softmax).

## mindspore.ops.log_softmax

```python
class mindspore.ops.log_softmax(
    logits,
    axis=-1,
)
```

For more information, see [mindspore.ops.log_softmax](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.log_softmax.html).

## Differences

PyTorch: Support the use of `dim` parameters and `input` input to implement functions that take the logits of the softmax result.

MindSpore: Support the use of `axis` parameters and `input` input to implement functions that take the logits of the softmax result.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | input  | logits    | Same function, different parameter names |
|      | Parameter 2 | dim  | axis | Same function, different parameter names |
|      | Parameter 3 | dtype | - | PyTorch is used to specify the data type of the output Tensor, which is not available in MindSpore. |

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, we can define an instance of this class first, and the default value of the parameter axis is -1.
x = ms.Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
output1 = ops.log_softmax(x)
print(output1)
# Out:
# [0.01165623 0.03168492 0.08612854 0.23412167 0.6364086 ]
x = ms.Tensor(np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), ms.float32)
output2 = ops.log_softmax(x, axis=0)
print(output2)
# out:
# [[0.01798621 0.11920292 0.5        0.880797   0.98201376]
#  [0.98201376 0.880797   0.5        0.11920292 0.01798621]]

# In torch, the input and dim should be input at the same time to implement the function.
input = torch.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
output3 = torch.nn.functional.log_softmax(input, dim=0)
print(output3)
# Out:
# tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364], dtype=torch.float64)
```
