# 比较与torch.nn.functional.adaptive_avg_pool2d的功能差异

## torch.nn.functional.adaptive_avg_pool2d

```python
torch.nn.functional.adaptive_avg_pool2d(
    input,
    output_size
)
```

## mindspore.nn.AvgPool2d

```python
class mindspore.nn.AvgPool2d(
    kernel_size=1,
    stride=1,
    pad_mode='valid',
    data_format='NCHW'
)(input)
```

## 使用方式

PyTorch: 对输入数据的H与W维执行平均池化。使用上，仅需指定池化后数据H和W维的期望形状即可。无需用户手工计算并指定`kernel_size`、`stride`等。

MindSpore：需用户手工计算并指定`kernel_size`、`stride`等。

## 代码示例

```python
import mindspore
from mindspore import Tensor, nn
import torch
import numpy as np

x = np.random.randint(0, 10, [1, 2, 4, 4])

# In MindSpore, parameters kernel_size and stride should be calculated in advance and set for pooling.
pool = nn.AvgPool2d(kernel_size=3, stride=1)
input_x = Tensor(x, mindspore.float32)
output = pool(input_x)
print(output.shape)
# Out：
# (1, 2, 2, 2)

# In torch, the shape of output can be set directly for pooling.
input_x = torch.tensor(x.astype(np.float32))
output = torch.nn.functional.adaptive_avg_pool2d(input_x, (2, 2))
print(output.shape)
# Out：
# torch.Size([1, 2, 2, 2])
```
