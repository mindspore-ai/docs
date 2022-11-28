# 比较与torch.nn.functional.pad的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Pad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.pad

```python
def torch.nn.functional.pad(
    input,
    pad,
    mode='constant',
    value=0
)
```

更多内容详见[torch.nn.functional.pad](https://pytorch.org/docs/1.8.1/nn.functional.html#pad)。

## mindspore.ops.pad

```python
def mindspore.ops.pad(
    input_x,
    padding,
    mode='constant',
    value=None
)
```

更多内容详见[mindspore.ops.pad](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.pad.html)。

## 使用方式

PyTorch：pad参数是一个有m个值的tuple，m/2小于等于输入数据的维度，且m为偶数。支持填充负维度。假设pad=(k1, k2, ..., kl, km)，输入x的shape为(d1, d2..., dg)，则dg维的两边分别填充长度为k1，k2的值。依此类推，d1维的两边分别填充长度为kl，km的值。

MindSpore：MindSpore的padding参数与PyTorch的pad参数功能用法完全一致，另外MindSpore相比PyTorch额外支持了Tensor类型的入参形式。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | input_x   | 功能一致，参数名不同 |
|      | 参数2 | pad     | padding   | 功能一致，参数名不同 |
|      | 参数3 | mode    | mode   | 功能一致，mindspore暂缺失circular模式 |
|      | 参数4 | value   | value   | 功能一致，constant模式下mindspore入参None的时候默认值为0 |

## 代码示例

```python
# In MindSpore.
import numpy as np
import torch
import mindspore.ops as ops
import mindspore as ms

x = ms.Tensor(np.ones([1, 2, 2, 3]).astype(np.float32))
padding = (1, 1, 2, 2)
output = ops.pad(x, padding)
print(output.shape)
# Out:
# (1, 2, 6, 5)

# In Pytorch.
x = torch.empty(1, 2, 2, 3)
pad = (1, 1, 2, 2)
output = torch.nn.functional.pad(x, pad)
print(output.size())
# Out:
# torch.Size([1, 2, 6, 5])
```
