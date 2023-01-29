# 比较与torch.conj的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/conj.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.conj

```text
torch.conj(input, *, out=None) -> Tensor
```

更多内容详见[torch.conj](https://pytorch.org/docs/1.8.1/generated/torch.conj.html)。

## mindspore.ops.conj

```text
mindspore.ops.conj(input) -> Tensor
```

更多内容详见[mindspore.ops.conj](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.conj.html)。

## 差异对比

PyTorch：返回输入张量的共轭复数。

MindSpore：MindSpore此API实现功能与PyTorch一致。

| 分类 | 子类  | PyTorch | MindSpore | 差异 |
| ---- | ----- | ------- | --------- | ---- |
| 参数 | 参数1 | input   | input     | -    |
|  | 参数2 | out   | -     | 不涉及    |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

a = tensor([-1 + 1j, -2 + 2j, 3 - 3j], dtype=torch.complex64)
b = torch.conj(a)
print(b.detach().numpy())
# [-1.-1.j -2.-2.j 3.+3.j]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

a = Tensor([-1 + 1j, -2 + 2j, 3 - 3j], dtype=mindspore.complex64)
b = ops.conj(a)
print(b)
# [-1.-1.j -2.-2.j 3.+3.j]
```