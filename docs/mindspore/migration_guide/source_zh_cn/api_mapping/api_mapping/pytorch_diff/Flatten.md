# 比较与torch.flatten的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Flatten.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.flatten

```python
torch.flatten(
    input,
    start_dim=0,
    end_dim=-1
)
```

## mindspore.ops.Flatten

```python
class mindspore.ops.Flatten(*args, **kwargs)(input_x)
```

## 使用方式

PyTorch: 支持指定维度对元素进行展开。

MindSpore：仅支持保留第0维元素，对其余维度的元素进行展开。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only the 0th dimension will be reserved and the rest will be flattened.
input_tensor = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
flatten = ops.Flatten()
output = flatten(input_tensor)
print(output.shape)
# Out：
# (1, 24)

# In torch, the dimension to reserve will be specified and the rest will be flattened.
input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output1 = torch.flatten(input=input_tensor, start_dim=1)
print(output1.shape)
# Out：
# torch.Size([1, 24])

input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output2 = torch.flatten(input=input_tensor, start_dim=2)
print(output2.shape)
# Out：
# torch.Size([1, 2, 12])
```