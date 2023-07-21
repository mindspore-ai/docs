# 比较与torch.min的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/min.md)

## torch.min

```python
torch.min(input, dim, keepdim=False, *, out=None)
```

更多内容详见[torch.min](https://pytorch.org/docs/1.8.1/torch.html#torch.min)。

## mindspore.ops.min

```python
mindspore.ops.min(input, axis=None, keepdims=False, *, initial=None, where=None)
```

更多内容详见[mindspore.ops.min](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.min.html)。

## 差异对比

PyTorch：输出为元组(最大值, 最大值的索引)。

MindSpore：axis为None或者shape为空时，keepdims以及后面的参数均不生效，功能与torch.min(input)一致，此时索引固定返回0；否则，输出为元组(最大值, 最大值的索引)，功能与torch.min(input, dim, keepdim=False, *, out=None)一致。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | input | 一致 |
| | 参数2 | dim | axis | 功能一致，参数名不同|
| | 参数3 | keepdim    | keepdims     | 功能一致，参数名不同       |
| | 参数4 | -      |initial    | 不涉及        |
| | 参数5 |  -     |where    | 不涉及        |
| | 参数6 | out    | -         | 不涉及 |

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

np_x = np.array([[-0.0081, -0.3283, -0.7814, -0.0934],
                 [1.4201, -0.3566, -0.3848, -0.1608],
                 [-0.0446, -0.1843, -1.1348, 0.5722],
                 [-0.6668, -0.2368, 0.2790, 0.0453]]).astype(np.float32)
# mindspore
input_x = ms.Tensor(np_x)
output, index = ops.min(input_x, axis=1)
print(output)
# [-0.7814 -0.3848 -1.1348 -0.6668]
print(index)
# [2 2 2 0]

# torch
input_x = torch.tensor(np_x)
output, index = torch.min(input_x, dim=1)
print(output)
# tensor([-0.7814, -0.3848, -1.1348, -0.6668])
print(index)
# tensor([2, 2, 2, 0])
```
