# 比较与torch.max的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/max.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

## torch.max

```python
torch.max(input, dim, keepdim=False, *, out=None)
```

更多内容详见[torch.max](https://pytorch.org/docs/1.8.1/torch.html#torch.max)。

## mindspore.ops.max

```python
mindspore.ops.max(input, axis=None, keepdims=False, *, initial=None, where=None)
```

更多内容详见[mindspore.ops.max](https://mindspore.cn/docs/zh-CN/r1.11/api_python/ops/mindspore.ops.max.html)。

## 差异对比

PyTorch：输出为元组(最大值, 最大值的索引)。

MindSpore：axis为None或者shape为空时，keepdims以及后面的参数均不生效，功能与torch.max(input)一致，此时索引固定返回0；否则，输出为元组(最大值, 最大值的索引)，功能与torch.max(input, dim, keepdim=False, *, out=None)一致。

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
output, index = ops.max(input_x, axis=1)
print(output)
# [-0.0081  1.4201  0.5722  0.279 ]
print(index)
# [0 0 3 2]

# torch
input_x = torch.tensor(np_x)
output, index = torch.max(input_x, dim=1)
print(output)
# tensor([-0.0081,  1.4201,  0.5722,  0.2790])
print(index)
# tensor([0, 0, 3, 2])
```
