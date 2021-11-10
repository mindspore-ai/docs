# 比较与torch.min的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/ArgMinWithValue.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.min

```python
torch.min(
    input,
    dim,
    keepdim=False,
    out=None
)
```

更多内容详见[torch.min](https://pytorch.org/docs/1.5.0/torch.html#torch.min)。

## mindspore.ops.ArgMinWithValue

```python
class mindspore.ops.ArgMinWithValue(
    axis=0,
    keep_dims=False
)(input_x)
```

更多内容详见[mindspore.ops.ArgMinWithValue](https://mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ArgMinWithValue.html#mindspore.ops.ArgMinWithValue)。

## 使用方式

PyTorch：输出为元组(最小值, 最小值的索引)。

MindSpore：输出为元组(最小值的索引, 最小值)。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# Output tuple(index of min, min).
input_x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
argmin = ops.ArgMinWithValue()
index, output = argmin(input_x)
print(index)
print(output)
# Out：
# 0
# 0.0

# Output tuple(min, index of min).
input_x = torch.tensor([0.0, 0.4, 0.6, 0.7, 0.1])
output, index = torch.min(input_x, 0)
print(index)
print(output)
# Out：
# tensor(0)
# tensor(0.)
```

