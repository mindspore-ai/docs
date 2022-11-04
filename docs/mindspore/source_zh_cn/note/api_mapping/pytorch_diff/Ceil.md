# 比较与torch.ceil的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Ceil.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.ceil

```text
torch.ceil(input) -> Tensor
```

更多内容详见 [torch.ceil](https://pytorch.org/docs/1.8.1/generated/torch.ceil.html)。

## mindspore.ops.ceil

```text
mindspore.ops.ceil(x) -> Tensor
```

更多内容详见 [mindspore.ops.ceil](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ceil.html)。

## 差异对比

PyTorch：返回带有 input 元素的 ceil 的新张量，该元素大于或等于每个元素的最小整数。

MindSpore: MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同 |

### 代码示例1

两API实功能一致，用法相同。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([2.5, -1.5, 1, -0.6448, 0.5826]), dtype=torch.float32)
output = torch.ceil(input).numpy()
print(output)
# [ 3. -1.  1. -0.  1.]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([2.5, -1.5, 1, -0.6448, 0.5826]), mindspore.float32)
output = ops.ceil(x).asnumpy()
print(output)
# [ 3. -1.  1. -0.  1.]
```
