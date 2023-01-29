# Function Differences with torch.float_power

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/float_power.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## torch.float_power

```python
torch.float_power(input, exponent, *, out=None) → Tensor
```

For more information, see [torch.float_power](https://pytorch.org/docs/1.8.1/generated/torch.float_power.html).

## mindspore.ops.float_power

```python
mindspore.ops.float_power(x, exponent)
```

For more information, see [mindspore.ops.float_power](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/ops/mindspore.ops.float_power.html#mindspore.ops.float_power).

## Differences

PyTorch: Raises input to the power of exponent, elementwise, in double precision. If neither input is complex returns a torch.float64 tensor, and if one or more inputs is complex returns a torch.complex128 tensor.

MindSpore:

- If neither input is complex Tensor, MindSpore’s API implements the same function as PyTorch, only the parameter names are different;
- If one or more inputs is complex Tensor, MindSpore will not improve the precision. Currently, complex number operations only support CPU;
    - When the input is two complex Tensors, MindSpore requires the two Tensors to be of the same type, and the return value is the same as the input type;
    - When the input is a complex Tensor and a scalar, the return value of MindSpore is the same type as the input Tensor;
    - When the input is a complex Tensor and a real Tensor, MindSpore currently does not support this operation.

| Categories | Subcategories | PyTorch | MindSpore | Differences       |
| ---- | ----- | ------- | --------- | -------------------- |
|Parameter | Parameter 1 | input   | x | The function is the same, and the parameter name is different |
|      | Parameter 2 | exponent | exponent | The function is the same |
|      | Parameter 3 | out     | -         | MindSpore does not have this Parameter      |

## Code Example1

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

## Code Example2

> When the input is of complex type, MindSpore only supports two complex Tensors or one complex Tensor and one scalar as the input on the CPU platform, and the return value type is the same as the input complex Tensor type.

```python
import numpy as np
input_np = np.array([(2., 3.), (3., 4.), (4., 5.)], np.complex64)
# PyTorch
import torch
input = torch.from_numpy(input_np)
out_torch = torch.float_power(input, 2.)
print(out_torch.detach().numpy(), out_torch.detach().numpy().dtype)
# [[ 4.+0.j  9.+0.j]
#  [ 9.+0.j 16.+0.j]
#  [16.+0.j 25.+0.j]] complex128

# MindSpore
import mindspore
from mindspore import Tensor, ops
x = Tensor(input_np)
output = ops.float_power(x, 2.)
print(output.asnumpy())
# [[ 4.      +0.j  9.      +0.j]
#  [ 9.      +0.j 16.      +0.j]
#  [16.      +0.j 25.000002+0.j]] complex64
```
