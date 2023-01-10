# Function Differences with torch.add

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/add.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.add

```text
torch.add(input, other, alpha=1) -> Tensor
```

For more information, see [torch.add](https://pytorch.org/docs/1.8.1/generated/torch.add.html).

## mindspore.ops.add

```text
mindspore.ops.add(x, y) -> Tensor
```

For more information, see [mindspore.ops.add](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.add.html).

## Differences

PyTorch: When the alpha parameter is not set, input and other are added element-wise, while when the parameter alpha is set, each element of the tensor other is multiplied by the scalar alpha and added to each element of the tensor input, to return the resultant tensor.

MindSpore: MindSpore implements the same function as PyTorch without the alpha parameter, and only the parameter name is different. MindSpore does not have this parameter.

| Categories | Subcategories   | PyTorch     | MindSpore   | Differences   |
| ---- | ----- | ------- | --------- | --------------------- |
| Parameters | Parameter 1 | input   | x         | Same function, different parameter names                    |
|      | Parameter 2 | other   | y         | Same function, different parameter names                    |
|      | Parameter 3 | alpha   | -         | The scalar multiplier of input other. MindSpore does not have this parameter |

### Code Example 1

When torch.add does not set the alpha parameter, the two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

input = torch.tensor([1, 2, 3], dtype=torch.float32)
other = torch.tensor([4, 5, 6], dtype=torch.float32)
out = torch.add(input, other).numpy()
print(out)
# [5. 7. 9.]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([1, 2, 3]).astype(np.float32))
y = Tensor(np.array([4, 5, 6]).astype(np.float32))
output = ops.add(x, y).asnumpy()
print(output)
# [5. 7. 9.]
```

### Code Example 2

When torch.add sets the alpha parameter, MindSpore uses the same alpha value to multiply by y before calling the add interface, which achieves the same effect as PyTorch.

```python
# PyTorch
import torch
from torch import tensor

input = torch.tensor([1, 2, 3], dtype=torch.float32)
other = torch.tensor([[1],[2],[3]], dtype=torch.float32)
out = torch.add(input, other, alpha=10).numpy()
print(out)
# [[11. 12. 13.]
#  [21. 22. 23.]
#  [31. 32. 33.]]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([1, 2, 3]).astype(np.float32))
y = Tensor(np.array([[1],[2],[3]]).astype(np.float32))
alpha = 10
output = ops.add(x, y * alpha).asnumpy()
print(output)
# [[11. 12. 13.]
#  [21. 22. 23.]
#  [31. 32. 33.]]
```
