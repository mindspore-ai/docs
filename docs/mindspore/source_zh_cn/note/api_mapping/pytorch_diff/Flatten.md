# 比较与torch.flatten的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Flatten.md)

## torch.flatten

```python
torch.flatten(
    input,
    start_dim=0,
    end_dim=-1
)
```

更多内容详见[torch.flatten](https://pytorch.org/docs/1.5.0/torch.html#torch.flatten)。

## mindspore.ops.Flatten

```python
class mindspore.ops.Flatten(*args, **kwargs)(input_x)
```

更多内容详见[mindspore.ops.Flatten](https://mindspore.cn/docs/zh-CN/r1.9/api_python/ops/mindspore.ops.Flatten.html#mindspore.ops.Flatten)。

## 使用方式

PyTorch：支持指定维度对元素进行展开。

MindSpore：仅支持保留第0维元素，对其余维度的元素进行展开。

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only the 0th dimension will be reserved and the rest will be flattened.
input_tensor = ms.Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
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