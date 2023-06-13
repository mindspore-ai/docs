# 比较与torch.nn.RNNCell的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RNNCell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

## torch.nn.RNNCell

```text
class torch.nn.RNNCell(
    input_size,
    hidden_size,
    bias=True,
    nonlinearity='tanh')(input, hidden) -> Tensor
```

更多内容详见[torch.nn.RNNCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNNCell.html)。

## mindspore.nn.RNNCell

```text
class mindspore.nn.RNNCell(
    input_size: int,
    hidden_size: int,
    has_bias: bool=True,
    nonlinearity: str = 'tanh')(x, hx) -> Tensor
```

更多内容详见[mindspore.nn.RNNCell](https://www.mindspore.cn/docs/zh-CN/r1.11/api_python/nn/mindspore.nn.RNNCell.html)。

## 差异对比

PyTorch：循环神经网络单元。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input_size | input_size |- |
| | 参数2 | hidden_size | hidden_size | - |
| | 参数3 | bias | has_bias | 功能一致，参数名不同 |
| | 参数4 | nonlinearity | nonlinearity | - |
|输入 | 输入1 | input | x | 功能一致，参数名不同 |
| | 输入2 | hidden | hx |  功能一致，参数名不同 |

### 代码示例1

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

rnncell = torch.nn.RNNCell(2, 3, nonlinearity="relu", bias=False)
input = torch.tensor(np.array([[3.0, 4.0]]).astype(np.float32))
hidden = torch.tensor(np.array([[1.0, 2.0, 3]]).astype(np.float32))
output = rnncell(input, hidden)
print(output)
# tensor([[0.5022, 0.0000, 1.4989]], grad_fn=<ReluBackward0>)

# MindSpore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

rnncell = nn.RNNCell(2, 3, nonlinearity="relu", has_bias=False)
x = Tensor(np.array([[3.0, 4.0]]).astype(np.float32))
hx = Tensor(np.array([[1.0, 2.0, 3]]).astype(np.float32))
output = rnncell(x, hx)
print(output)
# [[2.4998584 0.        1.9334991]]
```
