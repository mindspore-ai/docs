# 比较与torch.cos的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/cos.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.cos

``` text
torch.cos(input, *, out=None) -> Tensor
```

更多内容详见[torch.cos](https://pytorch.org/docs/1.8.1/generated/torch.cos.html)。

## mindspore.ops.cos

```text
mindspore.ops.cos(x) -> Tensor
```

更多内容详见[mindspore.ops.cos](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cos.html)。

## 差异对比

PyTorch：返回一个新的张量，内容为输入元素的余弦。

MindSpore：功能一致，仅参数名不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                     |
| ---- | ----- | ------- | --------- | ---------------------------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同                    |
|      | 参数2 | out     | -         | 不涉及 |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# 样例1 -> 代码示例1
# PyTorch
import torch

a = torch.tensor([0.24, 0.83, 0.31, 0.09],dtype=torch.float32)
output = torch.cos(a)
output = output.detach().numpy()
print(output)
# [0.971338   0.6748758  0.95233357 0.9959527 ]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
output = ops.cos(x)
print(output)
# [0.971338 0.6748758 0.95233357 0.9959527 ]
```
