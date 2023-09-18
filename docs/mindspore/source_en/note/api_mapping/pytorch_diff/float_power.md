# Differences with torch.float_power

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/float_power.md)

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.float_power    |   mindspore.ops.float_power    |
|    torch.Tensor.float_power   |  mindspore.Tensor.float_power   |

## torch.float_power

```python
torch.float_power(input, exponent, *, out=None) -> Tensor
```

For more information, see [torch.float_power](https://pytorch.org/docs/1.8.1/generated/torch.float_power.html).

## mindspore.ops.float_power

```python
mindspore.ops.float_power(input, exponent)
```

For more information, see [mindspore.ops.float_power](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.float_power.html#mindspore.ops.float_power).

## Differences

PyTorch: Raise the input tensor to double precision to calculate exponential powers. If neither input is complex, a torch.float64 tensor is returned, and if one or more inputs is complex, a torch.complex128 tensor is returned.

MindSpore: If the inputs are all real numbers, MindSpore API implements the same functionality as PyTorch, and only the parameter names are different. Currently, MindSpore does not support computation with complex numbers.

| Categories | Subcategories | PyTorch | MindSpore | Differences       |
| ---- | ----- | ------- | --------- | -------------------- |
|Parameter | Parameter 1 | input   | input | The function is the same |
|      | Parameter 2 | exponent | exponent | The function is the same |
|      | Parameter 3 | out     | -         | MindSpore does not have this Parameter      |

## Code Example

> When the input is a real number type, the functions of the two APIs are the same, and the usage is the same.

```python
import numpy as np
input_np = np.array([2., 3., 4.], np.float32)
# PyTorch
import torch
input = torch.from_numpy(input_np)
out_torch = torch.float_power(input, 2.)
print(out_torch.detach().numpy())
# [ 4.  9. 16.]

# MindSpore
import mindspore
from mindspore import Tensor, ops
x = Tensor(input_np)
output = ops.float_power(x, 2.)
print(output.asnumpy())
# [ 4.  9. 16.]
```
