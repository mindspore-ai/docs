# 比较与torch.nn.GRUCell的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/GRUCell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.GRUCell

```text
class torch.nn.GRUCell(
    input_size,
    hidden_size,
    bias=True)(input, hidden) -> Tensor
```

更多内容详见[torch.nn.GRUCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.GRUCell.html)。

## mindspore.nn.GRUCell

```text
class mindspore.nn.GRUCell(
    input_size: int,
    hidden_size: int,
    has_bias: bool=True)(x, hx) -> Tensor
```

更多内容详见[mindspore.nn.GRUCell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.GRUCell.html)。

## 差异对比

PyTorch：循环神经网络单元。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input_size | input_size |- |
| | 参数2 | hidden_size | hidden_size | - |
| | 参数3 | bias | has_bias | 功能一致，参数名不同 |
|输入 | 输入1 | input | x | 功能一致，参数名不同 |
| | 输入2 | hidden | hx |  功能一致，参数名不同 |

### 代码示例1

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

grucell = torch.nn.GRUCell(2, 3, bias=False)
input = torch.tensor(np.array([[3.0, 4.0]]).astype(np.float32))
hidden = torch.tensor(np.array([[1.0, 2.0, 3]]).astype(np.float32))
output = grucell(input, hidden)
print(output)
# tensor([[ 0.9948,  0.0913, -0.1633]], grad_fn=<AddBackward0>)

# MindSpore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

grucell = nn.GRUCell(2, 3, has_bias=False)
x = Tensor(np.array([[3.0, 4.0]]).astype(np.float32))
hx = Tensor(np.array([[1.0, 2.0, 3]]).astype(np.float32))
output = grucell(x, hx)
print(output)
# [[-0.94861907  0.6191679   2.1289415 ]]
```
