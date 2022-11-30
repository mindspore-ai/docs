# 比较与torch.not_equal的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/not_equal.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.not_equal

```text
torch.not_equal(input, other, *, out=None) -> Tensor
```

更多内容详见 [torch.not_equal](https://pytorch.org/docs/1.8.1/generated/torch.not_equal.html)。

## mindspore.ops.not_equal

```text
mindspore.ops.not_equal(x, other) -> Tensor
```

更多内容详见 [mindspore.ops.not_equal](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.not_equal.html)。

## 差异对比

PyTorch：逐元素计算 `input` 和 `other` 是否不相等。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名`input`不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异         |
| --- |-----|---------|-----------|------------|
|参数 | 参数1 | input   | x         | 功能一致，参数名不同 |
| | 参数2 | other   | other     | 功能一致       |
| | 参数3 | out     | -         | 不涉及        |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
output = torch.not_equal(input, other)
print(output.detach().numpy())
#[[False  True]
# [ True False]]

# MindSpore
import mindspore
from mindspore import Tensor, ops
import numpy as np
x = Tensor(np.array([[1, 2], [3, 4]]))
other = Tensor(np.array([[1, 1], [4, 4]]))
output = ops.not_equal(x, other)
print(output)
#[[False  True]
# [ True False]]
```
