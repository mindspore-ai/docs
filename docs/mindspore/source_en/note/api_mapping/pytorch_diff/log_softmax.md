# Function Differences with torch.nn.functional.log_softmax

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/log_softmax.md)

## torch.nn.functional.log_softmax

```python
torch.nn.functional.log_softmax(
    input,
    dim=None,
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

For more information, see [mindspore.ops.log_softmax](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.log_softmax.html).

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
# [-4.451912   -3.4519122  -2.4519122  -1.451912   -0.45191208]
x = ms.Tensor(np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), ms.float32)
output2 = ops.log_softmax(x, axis=0)
print(output2)
# out:
# [[-4.01815    -2.126928   -0.6931472  -0.12692805 -0.01814996]
#  [-0.01814996 -0.12692805 -0.6931472  -2.126928   -4.01815   ]]

# In torch, the input and dim should be input at the same time to implement the function.
input = torch.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
output3 = torch.nn.functional.log_softmax(input, dim=0)
print(output3)
# Out:
# tensor([-4.4519, -3.4519, -2.4519, -1.4519, -0.4519], dtype=torch.float64)
```
