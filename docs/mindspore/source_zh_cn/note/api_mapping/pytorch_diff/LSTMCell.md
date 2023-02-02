# 比较与torch.nn.LSTMCell的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/LSTMCell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.nn.LSTMCell

```text
class torch.nn.LSTMCell(
    input_size,
    hidden_size,
    bias=True)(input, h_0, c_0) -> Tensor
```

更多内容详见[torch.nn.LSTMCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.LSTMCell.html?torch.nn.LSTMCell)。

## mindspore.nn.LSTMCell

```text
class mindspore.nn.LSTMCell(
    input_size,
    hidden_size,
    has_bias=True)(x, hx) -> Tensor
```

更多内容详见[mindspore.nn.LSTMCell](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.LSTMCell.html)。

## 差异对比

PyTorch：计算长短期记忆网络单元。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，返回值在形式上有差异。PyTorch中返回h_1和 c_1，MindSpore中返回hx’，是两个Tensor组成的的元组(h’, c’)。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input_size | input_size |- |
| | 参数2 | hidden_size | hidden_size | - |
| | 参数3 | bias | has_bias | 功能一致，参数名不同 |
|输入 | 输入1 | input | x | 功能一致，参数名不同 |
| | 输入2 | h_0 | hx | 在MindSpore中hx表示两个Tensor(h_0, c_0)组成的元组，分别对应PyTorch中的输入2和3，功能相同  |
| | 输入3 | c_0 | hx | 在MindSpore中hx表示两个Tensor(h_0, c_0)组成的元组，分别对应PyTorch中的输入2和3，功能相同  |

### 代码示例1

> LSTMCell输入维度为10，隐藏状态维度为16，隐藏层为3行16列矩阵，cell为3行20列矩阵。5次for循环将整个序列依次计算，当前Cell的(hx,cx)作为下一次计算的隐藏层输入。两API实现相同的功能。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

rnn = torch.nn.LSTMCell(10, 16)
input = tensor(np.ones([5, 3, 10]).astype(np.float32))
hx = tensor(np.ones([3, 16]).astype(np.float32))
cx = tensor(np.ones([3, 16]).astype(np.float32))
output = []
for i in range(input.size()[0]):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)
print(tuple(output[0].shape))
# (3, 16)

# MindSpore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

net = nn.LSTMCell(10, 16)
x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
h = Tensor(np.ones([3, 16]).astype(np.float32))
c = Tensor(np.ones([3, 16]).astype(np.float32))
output = []
for i in range(5):
    hx = net(x[i], (h, c))
    output.append(hx)
print(output[0][0].shape)
# (3, 16)
```

### 代码示例2

> bias=False时无偏置b_ih和b_hh，该层不使用偏移权重，两API实现相同的功能。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

rnn = torch.nn.LSTMCell(10, 16, bias=False)
input = tensor(np.ones([5, 3, 10]).astype(np.float32))
hx = tensor(np.ones([3, 16]).astype(np.float32))
cx = tensor(np.ones([3, 16]).astype(np.float32))
output = []
for i in range(input.size()[0]):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)
print(tuple(output[0].shape))
# (3, 16)

# MindSpore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

net = nn.LSTMCell(10, 16, has_bias=False)
x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
h = Tensor(np.ones([3, 16]).astype(np.float32))
c = Tensor(np.ones([3, 16]).astype(np.float32))
output = []
for i in range(5):
    hx = net(x[i], (h, c))
    output.append(hx)
print(output[0][0].shape)
# (3, 16)
```
