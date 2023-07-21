# 比较与torch.clamp的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/clip_by_value.md)

## torch.clamp

```python
torch.clamp(
    input,
    min,
    max,
    out=None
)
```

更多内容详见[torch.clamp](https://pytorch.org/docs/1.5.0/torch.html#torch.clamp)。

## mindspore.ops.clip_by_value

```python
mindspore.ops.clip_by_value(
    x,
    clip_value_min,
    clip_value_max
)
```

更多内容详见[mindspore.ops.clip_by_value](https://mindspore.cn/docs/zh-CN/r1.9/api_python/ops/mindspore.ops.clip_by_value.html#mindspore.ops.clip_by_value)。

## 使用方式

PyTorch：将输入中的所有元素限制在 'min'、'max' 范围内并返回结果张量。 支持指定两个参数 'min'、'max' 之一。

MindSpore：将'x'的值限制在一个范围内，其下限为'clip_value_min'，上限为'clip_value_max'。 两个参数'clip_value_min'，'clip_value_max'是必要的。

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

min_value = ms.Tensor(5, ms.float32)
max_value = ms.Tensor(20, ms.float32)
x = ms.Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), ms.float32)
output = ops.clip_by_value(x, min_value, max_value)
print(output)
# Out：
# [[ 5. 20.  5.  7.]
#  [ 5. 11.  6. 20.]]

a = torch.randn(4)
print(a)
# Out：
#tensor([-1.7120,  0.1734, -0.0478, -0.0922])
print(torch.clamp(a, min=-0.5, max=0.5))
# Out：
# tensor([-0.5000,  0.1734, -0.0478, -0.0922])

a = torch.randn(4)
print(a)
# Out：
# tensor([-0.0299, -2.3184,  2.1593, -0.8883])
print(torch.clamp(a, min=0.5))
# Out：
# tensor([ 0.5000,  0.5000,  2.1593,  0.5000])

a = torch.randn(4)
print(a)
# Out：
# tensor([ 0.7753, -0.4702, -0.4599,  1.1899])
print(torch.clamp(a, max=0.5))
# Out：
# tensor([ 0.5000, -0.4702, -0.4599,  0.5000])
```