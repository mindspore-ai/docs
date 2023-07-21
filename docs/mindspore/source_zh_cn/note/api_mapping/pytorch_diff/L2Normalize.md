# 比较与torch.nn.functional.normalize的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/L2Normalize.md)

## torch.nn.functional.normalize

```python
torch.nn.functional.normalize(
    input,
    p=2,
    dim=1,
    eps=1e-12,
    out=None
)
```

更多内容详见[torch.nn.functional.normalize](https://pytorch.org/docs/1.5.0/nn.functional.html#torch.nn.functional.normalize)。

## mindspore.ops.L2Normalize

```python
class mindspore.ops.L2Normalize(
    axis=0,
    epsilon=1e-4
)(input_x)
```

更多内容详见[mindspore.ops.L2Normalize](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.L2Normalize.html#mindspore.ops.L2Normalize)。

## 使用方式

PyTorch：支持通过指定参数`p`来使用Lp范式。

MindSpore：仅支持L2范式。

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, you can directly pass data into the function, and the default dimension is 0.
l2_normalize = ops.L2Normalize()
input_x = ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
output = l2_normalize(input_x)
print(output)
# Out：
# [0.2673 0.5345 0.8018]

# In torch, parameter p should be set to determine it is a lp normalization, and the default dimension is 1.
input_x = torch.tensor(np.array([1.0, 2.0, 3.0]))
outputL2 = torch.nn.functional.normalize(input=input_x, p=2, dim=0)
outputL3 = torch.nn.functional.normalize(input=input_x, p=3, dim=0)
print(outputL2)
print(outputL3)
# Out：
# tensor([0.2673, 0.5345, 0.8018], dtype=torch.float64)
# tensor([0.3029, 0.6057, 0.9086], dtype=torch.float64)
```