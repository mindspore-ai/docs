# 比较与torch.nn.LSTM的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/LSTM.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.LSTM

```text
class torch.nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    bias=True,
    batch_first=False,
    dropout=0,
    bidirectional=False,
    proj_size=0)(input, (h_0, c_0)) -> Tensor
```

更多内容详见[torch.nn.LSTM](https://pytorch.org/docs/1.8.1/generated/torch.nn.LSTM.html)。

## mindspore.nn.LSTM

```text
class mindspore.nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    has_bias=True,
    batch_first=False,
    dropout=0,
    bidirectional=False)(x, hx, seq_length) -> Tensor
```

更多内容详见[mindspore.nn.LSTM](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.LSTM.html)。

## 差异对比

PyTorch：根据输入序列和给定的初始状态计算输出序列和最终状态。

MindSpore：若不指定PyTorch中的proj_size参数，MindSpore此API实现的功能与PyTorch一致，仅部分参数名不同。

| 分类 | 子类   | PyTorch       | MindSpore     | 差异                                                                            |
| --- |------|---------------|---------------|-------------------------------------------------------------------------------|
|参数 | 参数1  | input_size    | input_size    | -                                                                             |
| | 参数2  | hidden_size   | hidden_size   | -                                                                             |
| | 参数3  | num_layers    | num_layers    | -                                                                             |
| | 参数4  | bias          | has_bias      | 功能一致，参数名不同                                                                   |
| | 参数5  | batch_first   | batch_first   | -                                                                             |
| | 参数6  | dropout       | dropout       | -                                                                             |
| | 参数7  | bidirectional | bidirectional | -                                                                             |
| | 参数8  | proj_size     | -             | 在PyTorch中，若proj_size>0，输出shape中的hidden_size将会变成proj_size，默认值：0。MindSpore无此参数 |
| 输入 | 输入1 | input         | x             | 功能一致，参数名不同                                                                   |
| | 输入2 | h_0           | hx            | 在MindSpore中hx表示两个Tensor(h_0, c_0)组成的元组，分别对应PyTorch中的输入2和3，功能相同          |
| | 输入3 | c_0           | hx             | 在MindSpore中hx表示两个Tensor(h_0, c_0)组成的元组，分别对应PyTorch中的输入2和3，功能相同                                                                          |
| | 输入4 | -             | seq_length    | MindSpore中该参数可以指定输入batch的序列长度，PyTorch无此参数                                 |

### 代码示例

> 当PyTorch中的参数proj_size取默认值0时，两API实现的功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

rnn = torch.nn.LSTM(10, 16, 2, bias=True, batch_first=True, bidirectional=False)
input1 = tensor(np.ones([3, 5, 10]), dtype=torch.float32)
h0 = tensor(np.ones([1 * 2, 3, 16]), dtype=torch.float32)
c0 = tensor(np.ones([1 * 2, 3, 16]), dtype=torch.float32)
output, (hn, cn) = rnn(input1, (h0, c0))
print(output.detach().numpy().shape)
# (3, 5, 16)
print(hn.detach().numpy().shape)
# (2, 3, 16)
print(cn.detach().numpy().shape)
# (2, 3, 16)

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

net = mindspore.nn.LSTM(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
c0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
output, (hn, cn) = net(x, (h0, c0))
print(output.shape)
# (3, 5, 16)
print(hn.shape)
# (2, 3, 16)
print(cn.shape)
# (2, 3, 16)
```
