# 比较与torch.nn.GRU的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/GRU.md)

## torch.nn.GRU

```text
class torch.nn.GRU(
    input_size,
    hidden_size,
    num_layers=1,
    bias=True,
    batch_first=False,
    dropout=0,
    bidirectional=False)(input, h_0) -> Tensor
```

更多内容详见[torch.nn.GRU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GRU.html)。

## mindspore.nn.GRU

```text
class mindspore.nn.GRU(
    input_size,
    hidden_size,
    num_layers=1,
    has_bias=True,
    batch_first=False,
    dropout=0.0,
    bidirectional=False)(x, hx, seq_length) -> Tensor
```

更多内容详见[mindspore.nn.GRU](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.GRU.html)。

## 差异对比

PyTorch：根据输出序列和给定的初始状态计算输出序列和最终状态。

MindSpore：功能一致，多一个接口输入seq_length，表示输入batch中每个序列的长度。

| 分类 | 子类   | PyTorch       | MindSpore     | 差异                      |
| ---- | ------ | :------------ | ------------- | ------------------------- |
| 参数 | 参数1  | input_size    | input_size    | -                         |
|      | 参数2  | hidden_size   | hidden_size   | -                         |
|      | 参数3  | num_layers    | num_layers     | -                         |
|      | 参数4  | bias          | has_bias      | 功能一致，参数名不同      |
|      | 参数5  | batch_first   | batch_first   | -                         |
|      | 参数6  | dropout       | dropout       | -                         |
|      | 参数7  | bidirectional | bidirectional | -                         |
| 输入 | 输入1 | input         | x             | 功能一致，参数名不同      |
|      | 输入2 | h_0           | hx            | 功能一致，参数名不同      |
|      | 输入3 | -             | seq_length    | 输入batch中每个序列的长度 |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
import torch.nn as nn
import numpy as np

rnn = nn.GRU(10, 16, 2, batch_first=True)
input = torch.ones([3, 5, 10], dtype=torch.float32)
h0 = torch.ones([1 * 2, 3, 16], dtype=torch.float32)
output, hn = rnn(input, h0)
output = output.detach().numpy()
print(output.shape)
# (3, 5, 16)

# MindSpore
import mindspore
from mindspore import Tensor, nn

net = nn.GRU(10, 16, 2, batch_first=True)
x = Tensor(np.ones([3, 5, 10]), mindspore.float32)
h0 = Tensor(np.ones([1 * 2, 3, 16]), mindspore.float32)
output, hn = net(x, h0)
print(output.shape)
# (3, 5, 16)
```
