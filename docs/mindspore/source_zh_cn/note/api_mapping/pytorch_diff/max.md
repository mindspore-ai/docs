# 比较与torch.max的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/max.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.max

```python
torch.max(
    input,
    dim,
    keepdim=False,
    out=None
)
```

更多内容详见[torch.max](https://pytorch.org/docs/1.8.1/torch.html#torch.max)。

## mindspore.ops.max

```python
class mindspore.ops.max(
    x,
    axis=0,
    keep_dims=False
)
```

更多内容详见[mindspore.ops.max](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.max.html)。

## 使用方式

PyTorch：输出为元组(最大值, 最大值的索引)。

MindSpore：输出为元组(最大值的索引, 最大值)。

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# Output tuple(index of max, max).
input_x = ms.Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), ms.float32)
index, output = ops.max(input_x)
print(index)
# 3
print(output)
# 0.7

# Output tuple(max, index of max).
input_x = torch.tensor([0.0, 0.4, 0.6, 0.7, 0.1])
output, index = torch.max(input_x, 0)
print(index)
# tensor(3)
print(output)
# tensor(0.7000)
```