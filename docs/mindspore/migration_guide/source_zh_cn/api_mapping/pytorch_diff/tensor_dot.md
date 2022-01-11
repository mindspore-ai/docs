# 比较与torch.dot的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/tensor_dot.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.dot

```python
torch.dot(
    input,
    tensor
)
```

更多内容详见[torch.dot](https://pytorch.org/docs/1.5.0/torch.html#torch.dot)。

## mindspore.ops.tensor_dot

```python
mindspore.ops.tensor_dot(
    x1,
    x2,
    axes
)
```

更多内容详见[mindspore.ops.tensor_dot](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.tensor_dot.html#mindspore.ops.tensor_dot)。

## 使用方式

PyTorch：计算两个相同shape的tensor的点乘（内积），仅支持1D。

MindSpore：计算两个tensor在任意轴上的点乘，支持任意维度的tensor，但指定的轴对应的形状要相等。当输入为1D，轴设定为1时和PyTorch的功能一致。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, tensor of any dimension will be supported.
# And parameters will be set to specify how to compute among dimensions.
input_x1 = Tensor(np.array([2, 3, 4]), mindspore.float32)
input_x2 = Tensor(np.array([2, 1, 3]), mindspore.float32)
output = ops.tensor_dot(input_x1, input_x2, 1)
print(output)
# Out：
# 19.0

# In torch, only 1D tensor's computation will be supported.
input_x1 = torch.tensor([2, 3, 4])
input_x2 = torch.tensor([2, 1, 3])
output = torch.dot(input_x1, input_x2)
print(output)
# Out：
# tensor(19)
```