# 比较与torch.add的功能差异

## torch.add

```text
torch.add(input, other, alpha=1) -> Tensor
```

更多内容详见 [torch.add](https://pytorch.org/docs/1.8.1/generated/torch.add.html)。

## mindspore.ops.add

```text
mindspore.ops.add(x, y) -> Tensor
```

更多内容详见 [mindspore.ops.add](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.add.html)。

## 差异对比

PyTorch：不设置参数alpha时，输入input和输入other逐元素相加，设置参数alpha时，张量other的每个元素乘以标量alpha与张量input的每个逐元素相加，返回结果张量。

MindSpore：MindSpore此API实现功能与PyTorch不设置alpha参数时一致，仅参数名不同，MindSpore无此参数。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同                    |
|      | 参数2 | other   | y         | 功能一致，参数名不同                    |
|      | 参数3 | alpha   | -         | 输入other的标量乘数，MindSpore无此参数 |

### 代码示例1

torch.add不设置alpha参数时，两API实功能一致， 用法相同。

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

### 代码示例2

torch.add设置alpha参数时，MindSpore在调用add接口前，使用相同alpha值与y相乘，可以实现与PyTorch同样的效果。

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

