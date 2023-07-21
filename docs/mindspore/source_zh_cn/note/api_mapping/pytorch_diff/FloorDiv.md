# 比较与torch.floor_divide的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/FloorDiv.md)

## torch.floor_divide

```python
torch.floor_divide(
    input,
    other,
    out=None
)
```

更多内容详见[torch.floor_divide](https://pytorch.org/docs/1.5.0/torch.html#torch.floor_divide)。

## mindspore.ops.FloorDiv

```python
class mindspore.ops.FloorDiv(*args, **kwargs)(
    input_x,
    input_y
)
```

更多内容详见[mindspore.ops.FloorDiv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.FloorDiv.html#mindspore.ops.FloorDiv)。

## 使用方式

PyTorch：结果是往0方向取整，而非真的向下取整。例如相除为-0.9，取整后的结果为0。

MindSpore：结果按floor方式向下取整。例如相除为-0.9，取整后的结果为-1。

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, the output will be rounded toward the floor, so, after division, the output -0.33 will be rounded to -1.
input_x = ms.Tensor(np.array([2, 4, -1]), ms.int32)
input_y = ms.Tensor(np.array([3, 3, 3]), ms.int32)
floor_div = ops.FloorDiv()
output = floor_div(input_x, input_y)
print(output)
# Out：
# [ 0  1 -1]

# In torch, the output will be rounded toward 0, so, after division, the output -0.33 will be rounded to 0.
input_x = torch.tensor(np.array([2, 4, -1]))
input_y = torch.tensor(np.array([3, 3, 3]))
output = torch.floor_divide(input_x, input_y)
print(output)
# Out：
# tensor([0, 1, 0])
```