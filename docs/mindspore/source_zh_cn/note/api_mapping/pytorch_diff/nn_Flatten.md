# 比较与torch.nn.Flatten的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/nn_Flatten.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Flatten

```python
class torch.nn.Flatten(
    start_dim=1,
    end_dim=-1
)
```

更多内容详见[torch.nn.Flatten](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Flatten)。

## mindspore.nn.Flatten

```python
class mindspore.nn.Flatten()(input)
```

更多内容详见[mindspore.nn.Flatten](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.Flatten.html#mindspore.nn.Flatten)。

## 使用方式

PyTorch：支持指定维度对元素进行展开，默认保留第0维，对其余维度的元素进行展开；需要同`torch.nn.Sequential`一起使用。

MindSpore：仅支持保留第0维元素，对其余维度的元素进行展开。

## 代码示例

```python
import mindspore
from mindspore import Tensor, nn
import torch
import numpy as np

# In MindSpore, only the 0th dimension will be reserved and the rest will be flattened.
input_tensor = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
flatten = nn.Flatten()
output = flatten(input_tensor)
print(output.shape)
# Out：
# (1, 24)

# In torch, the dimension to reserve can be specified and the rest will be flattened.
# Different from torch.flatten, you should pass it as parameter into torch.nn.Sequential.
input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
flatten1 = torch.nn.Sequential(torch.nn.Flatten(start_dim=1))
output1 = flatten1(input_tensor)
print(output1.shape)
# Out：
# torch.Size([1, 24])

input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
flatten2 = torch.nn.Sequential(torch.nn.Flatten(start_dim=2))
output2 = flatten2(input_tensor)
print(output2.shape)
# Out：
# torch.Size([1, 2, 12])
```
