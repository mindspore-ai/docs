# 比较与torch.Tensor.float的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Cast.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.Tensor.float

```python
torch.Tensor.float()
```

## mindspore.ops.Cast

```python
class mindspore.ops.Cast(*args, **kwargs)(
    input_x,
    type
)
```

## 使用方式

PyTorch: 将tensor类型转成为float类型。

MindSpore：将输入类型转换为指定的数据类型。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, you can specify the data type to be transformed into.
input_x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
cast = ops.Cast()
output = cast(input_x, mindspore.int32)
print(output.dtype)
print(output.shape)
# Out：
# Int32
# (2, 3, 4, 5)

# In torch, the input will be transformed into float.
input_x = torch.Tensor(np.random.randn(2, 3, 4, 5).astype(np.int32))
output = input_x.float()
print(output.dtype)
print(output.shape)
# Out：
# torch.float32
# torch.Size([2, 3, 4, 5])
```