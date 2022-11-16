# 比较与torch.conj的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Conj.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.conj

``` text
torch.conj(input, *, out=None) -> Tensor
```

更多内容详见 [torch.conj](https://pytorch.org/docs/1.8.1/generated/torch.conj.html)。

## mindspore.ops.Conj

``` text
class mindspore.ops.Conj()(input) -> Tensor
```

更多内容详见 [mindspore.ops.Conj](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Conj.html)。

## 差异对比

Pytorch：返回输入张量的共轭复数。

MindSpore：MindSpore此API实现功能与PyTorch一致。

| 分类 | 子类  | Pytorch | MindSpore | 差异 |
| ---- | ----- | ------- | --------- | ---- |
| 参数 | 参数1 | input   | input     | -    |

### 代码示例

> 两API实功能一致， 用法相同。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

a = tensor([-1 + 1j, -2 + 2j, 3 - 3j], dtype=torch.complex64)
b = torch.conj(a)
print(a.detach().numpy())
print(b.detach().numpy())
# [-1.+1.j, -2.+2.j,  3.-3.j]
# [-1.-1.j, -2.-2.j,  3.+3.j]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

a = Tensor([-1 + 1j, -2 + 2j, 3 - 3j], dtype=mindspore.complex64)
conj = ops.Conj()
b = conj(a)
print(a)
print(b)
# [-1.+1.j, -2.+2.j,  3.-3.j]
# [-1.-1.j, -2.-2.j,  3.+3.j]

```