# 比较与torch.Tensor.float的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Cast.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.Tensor.float

```python
torch.Tensor.float(memory_format=torch.preserve_format)
```

更多内容详见[torch.Tensor.float](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.float)。

## mindspore.ops.Cast

```python
class mindspore.ops.Cast(*args, **kwargs)(
    input_x,
    type
)
```

更多内容详见[mindspore.ops.Cast](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Cast.html#mindspore.ops.Cast)。

## 使用方式

PyTorch：将tensor类型转成为float类型。

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