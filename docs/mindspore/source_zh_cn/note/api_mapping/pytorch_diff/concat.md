# 比较与torch.cat的功能差异

## torch.cat

```text
torch.cat(
    tensors,
    dim=0,
    out=None
) -> Tensor
```

更多内容详见 [torch.cat](https://pytorch.org/docs/1.8.1/generated/torch.cat.html)。

## mindspore.ops.concat

```text
mindspore.ops.concat(input_x, axis=0) -> Tensor
```

更多内容详见 [mindspore.ops.concat](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.concat.html)。

## 差异对比

PyTorch：在指定轴上拼接输入Tensor。输入tensor的数据类型不同时，低精度tensor会自动转成高精度tensor。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。当前要求输入tensor的数据类型保持一致，若不一致时可通过ops.cast把低精度tensor转成高精度类型再调用concat算子。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | tensors  | input_x | 功能一致， 参数名不同 |
|参数 | 参数1 | dim | axis |功能一致， 参数名不同 |
|  | 参数2  | out | - | 功能一致，MindSpore无此参数          |

### 代码示例1

> Mindspore当前要求输入tensor的数据类型保持一致，若不一致时可通过ops.cast把低精度tensor转成高精度类型再调用concat算子。

```python
# PyTorch
import torch

torch_x1 = torch.Tensor([[0, 1], [2, 3]]).type(torch.float32)
torch_x2 = torch.Tensor([[0, 1], [2, 3]]).type(torch.float32)
torch_x3 = torch.Tensor([[0, 1], [2, 3]]).type(torch.float16)

torch_output = torch.cat((torch_x1, torch_x2, torch_x3))
print(torch_output.numpy())
#[[0. 1.]
# [2. 3.]
# [0. 1.]
# [2. 3.]
# [0. 1.]
# [2. 3.]]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

# In MindSpore，converting low precision to high precision is needed before concat.
ms_x1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
ms_x2 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
ms_x3 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float16))

ms_x3 = mindspore.ops.cast(ms_x2, mindspore.float32)
output = mindspore.ops.concat((ms_x1, ms_x2, ms_x3))
print(output)
#[[0. 1.]
# [2. 3.]
# [0. 1.]
# [2. 3.]
# [0. 1.]
# [2. 3.]]
```
