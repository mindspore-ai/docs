# 比较与torch.nn.Tanh的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Tanh.md)

## torch.nn.Tanh

```text
class torch.nn.Tanh()(input) -> Tensor
```

更多内容详见[torch.nn.Tanh](https://pytorch.org/docs/1.8.1/generated/torch.nn.Tanh.html)。

## mindspore.nn.Tanh

```text
class mindspore.nn.Tanh()(x) -> Tensor
```

更多内容详见[mindspore.nn.Tanh](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/nn/mindspore.nn.Tanh.html)。

## 差异对比

PyTorch：计算双曲正切函数tanh。

MindSpore：MindSpore此API实现功能与PyTorch一致。

| 分类 | 子类  | PyTorch | MindSpore | 差异 |
| ---- | ----- | ------- | --------- | ---- |
| 输入 | 单输入 | input      | x         | 功能一致，参数名不同  |

### 代码示例

> 计算输入`x`的tanh函数，MindSpore此API功能与PyTorch一致。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor, nn

m = nn.Tanh()
x_ = np.array([0.7713, 0.0208, 0.6336], dtype=np.float32)
x = tensor(x_)
output = m(x)
print(output.numpy())
# [0.64768475 0.020797   0.56052613]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, nn

m = nn.Tanh()
x_ = np.array([0.7713, 0.0208, 0.6336], dtype=np.float32)
x = Tensor(x_, mindspore.float32)
output = m(x)
print(output)
# [0.64768475 0.020797   0.56052613]
```
