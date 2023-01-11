# Function Differences with torch.bmm

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BatchMatMul.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.bmm

```text
torch.bmm(input, mat2, *, deterministic=False, out=None) -> Tensor
```

For more information, see [torch.bmm](https://pytorch.org/docs/1.8.1/generated/torch.bmm.html).

## mindspore.ops.BatchMatMul

```text
mindspore.ops.BatchMatMul(transpose_a=False, transpose_b=False)(x, y) -> Tensor
```

For more information, see [mindspore.ops.BatchMatMul](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.BatchMatMul.html).

## Differences

PyTorch: Perform a batch matrix product on input and mat2, where input and mat2 must be 3-D tensors. If input is a (b, n, m) tensor and mat2 is a (b, n, p) tensor, the result of the product of the two matrices out is (b, n, p).

MindSpore: MindSpore API basically implements the same function as PyTorch, but MindSpore supports matrix multiplication in 3D and higher dimensions, where if MindSpore transpose_a is True, the last two dimensions of the first tensor of the input multiplication will be swapped.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | input         | x           | Same function, different parameter names                 |
|      | Parameter 2 | mat2         | y           | Same function, different parameter names                 |
|      | Parameter 3 | deterministic | -           | This parameter is only applicable to sparse sparse dense CUDA bmm. MindSpore does not have this parameter    |
|      | Parameter 4 | out    | -    | Not involved           |
|      | Parameter 5 | -    | transpose_a | If transpose_a is True, it will swap the last two dimensions of the first tensor of the input multiplication. |
|      | Parameter 6 | -    | transpose_b | If transpose_b is True, it will swap the last two dimensions of the second tensor of the input multiplication. |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.ones(shape=[2, 1, 5]), dtype=torch.float32)
mat2 = torch.tensor(np.ones(shape=[2, 5, 2]), dtype=torch.float32)
output = torch.bmm(input, mat2).numpy()
print(output)
# [[[5. 5.]]
#  [[5. 5.]]]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.ones(shape=[2, 1, 5]), mindspore.float32)
y = Tensor(np.ones(shape=[2, 5, 2]), mindspore.float32)

batmatmul = ops.BatchMatMul()
output = batmatmul(x, y)
print(output)
# [[[5. 5.]]
#  [[5. 5.]]]
```

### Code Example 2

PyTorch only supports 3D tensor, while MindSpore supports 3D and higher dimensional matrix multiplication calculations.

```python
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.ones(shape=[3, 5, 1, 3]), mindspore.float32)
y = Tensor(np.ones(shape=[3, 5, 3, 4]), mindspore.float32)

batmatmul = ops.BatchMatMul()
output = batmatmul(x, y)
print(output.shape)
# (3, 5, 1, 4)
```

### Code Example 3

If MindSpore transpose_a is True, it will swap the last two dimensions of the first tensor multiplied by the input, while transpose_b is True, it will swap the last two dimensions of the second tensor multiplied by the input.

```python
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.ones(shape=[3, 5, 3, 1]), mindspore.float32)
y = Tensor(np.ones(shape=[3, 5, 3, 4]), mindspore.float32)

batmatmul = ops.BatchMatMul(transpose_a=True)
output = batmatmul(x, y)
print(output.shape)
# (3, 5, 1, 4)
```
