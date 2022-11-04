# 比较与torch.asin的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Asin.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.asin

```text
torch.asin(input) -> Tensor
```

更多内容详见 [torch.asin](https://pytorch.org/docs/1.8.1/generated/torch.asin.html)。

## mindspore.ops.asin

```text
mindspore.ops.asin(x) -> Tensor
```

更多内容详见 [mindspore.ops.asin](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.asin.html)。

## 差异对比

PyTorch：逐元素计算输入Tensor的反正弦值。

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

input = torch.tensor(np.array([-0.5962, 1.04, 0.30, -0.4396]), dtype=torch.float32)
output = torch.asin(input).numpy()
print(output)
# [-0.6387595          nan  0.30469266 -0.4551533 ]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([-0.5962, 1.04, 0.30, -0.4396]), mindspore.float32)
output = ops.asin(x)
print(output)
# [-0.6387595          nan  0.30469266 -0.4551533 ]
```
