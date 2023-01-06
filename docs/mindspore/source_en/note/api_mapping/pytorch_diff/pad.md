# # Function Differences with torch.nn.functional.pad

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/pad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.pad

```python
torch.nn.functional.pad(
    input,
    pad,
    mode='constant',
    value=0
)
```

For more information, see [torch.nn.functional.pad](https://pytorch.org/docs/1.8.1/nn.functional.html#pad).

## mindspore.ops.pad

```python
mindspore.ops.pad(
    input_x,
    padding,
    mode='constant',
    value=None
)
```

For more information, see [mindspore.ops.pad](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.pad.html).

## Differences

PyTorch: The pad parameter is a tuple with m values, m/2 is less than or equal to the dimension of the input data, and m is even. Negative dimensions are supported. Assuming pad=(k1, k2, ..., kl, km), the shape of the input x is (d1, d2..., dg), then the two sides of the dg dimension are filled with the values of lengths k1 and k2 respectively. Similarly, the two sides of the d1 dimension are filled with the values of length kl and km respectively.

MindSpore: The function and usage of the padding parameter of MindSpore is completely consistent with that of the pad parameter of PyTorch. In addition, MindSpore supports the input form of Tensor type in addition to PyTorch.

| Classification | Subclass  | PyTorch | MindSpore | difference |
| ---- | ----- | ------- | --------- | -------------------- |
| parameter | parameter1 | input   | input_x   | Same functions, different parameter names |
|      | parameter2 | pad     | padding   | Same functions, different parameter names |
|      | parameter3 | mode    | mode   | The functions are consistent. MindSpore is temporarily missing the circular mode |
|      | parameter4 | value   | value   | The functions are consistent. In constant mode, the default value is 0 when MindSpore enters the parameter None |

## Code Example

```python
# In MindSpore.
import numpy as np
import torch
import mindspore.ops as ops
import mindspore as ms

x = ms.Tensor(np.ones([1, 2, 2, 3]).astype(np.float32))
padding = (1, 1, 2, 2)
output = ops.pad(x, padding)
print(output.shape)
# Out:
# (1, 2, 6, 5)

# In Pytorch.
x = torch.empty(1, 2, 2, 3)
pad = (1, 1, 2, 2)
output = torch.nn.functional.pad(x, pad)
print(output.size())
# Out:
# torch.Size([1, 2, 6, 5])
```
