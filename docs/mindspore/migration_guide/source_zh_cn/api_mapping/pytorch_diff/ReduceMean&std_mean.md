# 比较与torch.std_mean的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/ReduceMean&std_mean.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.std_mean

```python
torch.std_mean(
    input,
    dim,
    unbiased=True,
    keepdim=False
)
```

更多内容详见[torch.std_mean](https://pytorch.org/docs/1.5.0/torch.html#torch.std_mean)。

## mindspore.ops.ReduceMean

```python
class mindspore.ops.ReduceMean(keep_dims=False)(
    input_x,
    axis=()
)
```

更多内容详见[mindspore.ops.ReduceMean](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.ReduceMean.html#mindspore.ops.ReduceMean)。

## 使用方式

PyTorch：计算指定维度数据的标准差和平均值。

MindSpore：计算指定维度数据的平均值。

## 代码示例

```python
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only the mean of given dimension will be returned.
input_x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
op = ops.ReduceMean(keep_dims=True)
output = op(x=input_x, axis=1)
print(output)
# Out：
# [[1.5]
#  [3.5]]

# In torch, both std and mean of given dimensions will be returned.
input_x = torch.tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
output = torch.std_mean(input=input_x, dim=1)
std, mean = output
print('std: {}'.format(std))
print('mean: {}'.format(mean))
# Out：
# torch.tensor([0.7071, 0.7071])
# torch.tensor([1.5000, 3.5000])
```