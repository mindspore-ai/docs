# 比较与torch.acos的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/acos.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.acos

```text
torch.acos(input, *, out=None) -> Tensor
```

更多内容详见[torch.acos](https://pytorch.org/docs/1.8.1/generated/torch.acos.html)。

## mindspore.ops.acos

```text
mindspore.ops.acos(x) -> Tensor
```

更多内容详见[mindspore.ops.acos](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.acos.html)。

## 差异对比

PyTorch：逐元素计算输入Tensor的反余弦。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同 |
|      | 参数2 | out     | -         | 不涉及               |

### 代码示例1

两API实现功能一致，用法相同。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([0.74, 0.04, 0.30, 0.56]), dtype=torch.float32)
output = torch.acos(input).numpy()
print(output)
# [0.737726  1.5307857 1.2661036 0.9764105]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
output = ops.acos(x)
print(output)
# [0.737726  1.5307857 1.2661036 0.9764105]
```
