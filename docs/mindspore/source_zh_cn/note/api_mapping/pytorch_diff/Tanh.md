# 比较与torch.nn.Tanh的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Tanh.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Tanh

```text
class torch.nn.Tanh()(input) -> Tensor
```

更多内容详见[torch.nn.Tanh](https://pytorch.org/docs/1.8.1/generated/torch.nn.Tanh.html)。

## mindspore.nn.Tanh

```text
class mindspore.nn.Tanh()(x) -> Tensor
```

更多内容详见[mindspore.nn.Tanh](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Tanh.html)。

## 差异对比

PyTorch：计算双曲正切函数tanh。

MindSpore：MindSpore此API实现功能与PyTorch一致。

| 分类 | 子类  | PyTorch | MindSpore | 差异 |
| ---- | ----- | ------- | --------- | ---- |
| 参数 | 参数1 | input      | x         | 功能一致，参数名不同  |

### 代码示例1

> 计算输入`x`的tanh函数，MindSpore此API功能与PyTorch一致。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor, nn

m = nn.Tanh()
x_ = np.array([0.7713, 0.0208, 0.6336])
x = tensor(x_)
output = m(x)
print(output.numpy())
# [0.64768474 0.020797   0.56052611]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, nn

m = nn.Tanh()
x_ = np.array([0.7713, 0.0208, 0.6336])
x = Tensor(x_, mindspore.float32)
output = m(x)
print(output)
# [0.64768475 0.020797   0.56052613]
```
