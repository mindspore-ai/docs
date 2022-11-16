# 比较与torch.expm1的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/expm1.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.expm1

```text
torch.expm1(input) -> Tensor
```

更多内容详见 [torch.expm1](https://pytorch.org/docs/1.8.1/generated/torch.expm1.html)。

## mindspore.ops.expm1

```text
mindspore.ops.expm1(x) -> Tensor
```

更多内容详见 [mindspore.ops.expm1](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.expm1.html)。

## 差异对比

PyTorch：逐元素计算输入张量的指数减1的值

MindSpore：与PyTorch实现同样的功能，仅参数名不同

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | x |功能一致，参数名不同 |

### 代码示例1

> 两API实现功能相同，用法相同

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

inputx_ = np.array([0.0, 1.0, 2.0, 4.0])
inputx = tensor(inputx_, dtype=torch.float32)
output = torch.expm1(inputx)
output_m = output.detach().numpy()
print(output_m)
#[ 0.         1.7182817  6.389056  53.59815  ]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.array([0.0, 1.0, 2.0, 4.0])
x = Tensor(x_, mindspore.float32)
output = ops.expm1(x)
print(output)
#[0.        1.7182819  6.389056  53.598152]
```
